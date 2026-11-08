// nn::socket on Winsock (Windows counterpart of socket.cpp). Winsock keeps the BSD option values and
// error numbers (WSAExxx = 10000 + BSD errno), so most values pass through. Guest descriptors are
// small numbers mapped to either a SOCKET or an emulated eventfd (the SDK's internal plumbing waits
// on eventfds with poll). The game stays offline: name resolution always fails.
#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../native.h"

using namespace rt;

namespace {

thread_local s32 last_errno = 0;

struct EventFd {
    u64 counter = 0;
    bool semaphore = false, nonblocking = false;
};
struct Fd {
    SOCKET socket = INVALID_SOCKET;
    std::shared_ptr<EventFd> event;
};

std::mutex fds_mutex;
std::condition_variable events_changed; // any eventfd counter changed
std::unordered_map<s32, Fd> fds;
s32 next_fd = 3;

void InitWinsock() {
    static std::once_flag once;
    std::call_once(once, [] { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); });
}

s64 Fail(s32 bsd_errno) { last_errno = bsd_errno; return -1; }
s64 Wsa(s64 r) {
    if (r == SOCKET_ERROR) {
        const int e = WSAGetLastError();
        last_errno = e >= 10000 && e < 11000 ? e - 10000 : 22; // EINVAL
        return -1;
    }
    return r;
}

s32 Add(Fd f) {
    std::lock_guard lock{fds_mutex};
    const s32 fd = next_fd++;
    fds[fd] = std::move(f);
    return fd;
}
bool Get(s32 fd, Fd& out) {
    std::lock_guard lock{fds_mutex};
    auto it = fds.find(fd);
    if (it == fds.end()) return false;
    out = it->second;
    return true;
}
SOCKET Sock(Cpu* c, int i = 0) {
    Fd f;
    return Get((s32)c->x[i], f) && !f.event ? f.socket : INVALID_SOCKET;
}

// BSD sockaddr {u8 len; u8 family; data...} <-> Windows sockaddr {u16 family; data...}.
int ToHostAddr(const u8* guest, u32 len, sockaddr_storage& host) {
    std::memset(&host, 0, sizeof host);
    if (!guest || len < 2) return 0;
    const u8 family = guest[1];
    host.ss_family = family == 28 ? AF_INET6 : family;
    std::memcpy(reinterpret_cast<u8*>(&host) + 2, guest + 2, std::min<size_t>(len - 2, sizeof host - 2));
    return family == 28 ? (int)sizeof(sockaddr_in6) : (family == 2 ? (int)sizeof(sockaddr_in) : (int)len);
}
void ToGuestAddr(const sockaddr_storage& host, int host_len, u8* guest, u32* guest_len) {
    if (!guest || !guest_len) return;
    const u32 n = std::min<u32>(*guest_len, (u32)host_len);
    std::memcpy(guest, &host, n);
    if (n >= 2) {
        guest[0] = (u8)host_len;
        guest[1] = host.ss_family == AF_INET6 ? 28 : (u8)host.ss_family;
    }
    *guest_len = (u32)host_len;
}
int HostMsgFlags(int f) { return f & (MSG_OOB | MSG_PEEK | MSG_DONTROUTE | MSG_WAITALL); }

bool HostOption(int level, int opt) {
    if (level == 0xffff) // SOL_SOCKET: same numbering; no SO_REUSEPORT, and timeouts use another format
        return opt != 0x200 && opt != 0x1005 && opt != 0x1006;
    return level == IPPROTO_TCP || level == IPPROTO_IP || level == IPPROTO_IPV6;
}

// BSD pollfd {int fd; short events; short revents}; POLLIN 1, POLLPRI 2, POLLOUT 4, ERR 8, HUP 0x10.
struct GuestPollFd { s32 fd; s16 events, revents; };
s64 Poll(GuestPollFd* p, u64 count, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms);
    for (;;) {
        int ready = 0;
        std::vector<WSAPOLLFD> sockets;
        std::vector<u64> socket_index;
        {
            std::lock_guard lock{fds_mutex};
            for (u64 i = 0; i < count; ++i) {
                p[i].revents = 0;
                auto it = fds.find(p[i].fd);
                if (p[i].fd < 0) continue;
                if (it == fds.end()) { p[i].revents = 0x20; ++ready; continue; } // POLLNVAL
                if (auto& e = it->second.event) {
                    if ((p[i].events & 1) && e->counter) p[i].revents |= 1;
                    if (p[i].events & 4) p[i].revents |= 4;
                    if (p[i].revents) ++ready;
                } else {
                    WSAPOLLFD w{};
                    w.fd = it->second.socket;
                    if (p[i].events & 1) w.events |= POLLRDNORM;
                    if (p[i].events & 4) w.events |= POLLWRNORM;
                    sockets.push_back(w);
                    socket_index.push_back(i);
                }
            }
        }
        if (!sockets.empty()) {
            const int r = WSAPoll(sockets.data(), (ULONG)sockets.size(), 0);
            if (r > 0)
                for (size_t k = 0; k < sockets.size(); ++k) {
                    s16 rev = 0;
                    if (sockets[k].revents & (POLLRDNORM | POLLRDBAND)) rev |= 1;
                    if (sockets[k].revents & POLLWRNORM) rev |= 4;
                    if (sockets[k].revents & POLLERR) rev |= 8;
                    if (sockets[k].revents & POLLHUP) rev |= 0x10;
                    if (rev) { p[socket_index[k]].revents = rev; ++ready; }
                }
        }
        if (ready || timeout_ms == 0 || (timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline)) return ready;
        // Sleep until an eventfd changes (or briefly, when sockets are polled too).
        std::unique_lock lock{fds_mutex};
        auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(sockets.empty() ? 50 : 5);
        if (timeout_ms > 0) until = std::min(until, deadline);
        events_changed.wait_until(lock, until);
    }
}

} // namespace

NATIVE("nnsocketSocket") {
    InitWinsock();
    const int family = (int)c->x[0] == 28 ? AF_INET6 : (int)c->x[0];
    const SOCKET s = socket(family, (int)c->x[1] & 0xffff, (int)c->x[2]);
    if (s == INVALID_SOCKET) return Ret(c, (u64)Wsa(SOCKET_ERROR));
    if ((int)c->x[1] & 0x20000000) { u_long on = 1; ioctlsocket(s, FIONBIO, &on); }
    Ret(c, (u64)(s64)Add({s, nullptr}));
}
NATIVE("nnsocketClose") {
    Fd f;
    {
        std::lock_guard lock{fds_mutex};
        auto it = fds.find((s32)c->x[0]);
        if (it == fds.end()) return Ret(c, (u64)Fail(9)); // EBADF
        f = it->second;
        fds.erase(it);
    }
    if (!f.event) closesocket(f.socket);
    Ret(c, 0);
}
NATIVE("nnsocketShutdown") { Ret(c, (u64)Wsa(shutdown(Sock(c), (int)c->x[1]))); }
NATIVE("nnsocketListen") { Ret(c, (u64)Wsa(listen(Sock(c), (int)c->x[1]))); }
NATIVE("nnsocketBind") {
    sockaddr_storage a;
    const int len = ToHostAddr(Ptr<u8>(c, 1), (u32)c->x[2], a);
    Ret(c, (u64)Wsa(bind(Sock(c), reinterpret_cast<sockaddr*>(&a), len)));
}
NATIVE("nnsocketConnect") {
    sockaddr_storage a;
    const int len = ToHostAddr(Ptr<u8>(c, 1), (u32)c->x[2], a);
    const SOCKET s = Sock(c);
    rt::CoreRelease release;
    Ret(c, (u64)Wsa(connect(s, reinterpret_cast<sockaddr*>(&a), len)));
}
NATIVE("nnsocketAccept") {
    sockaddr_storage a;
    int len = sizeof a;
    const SOCKET listener = Sock(c);
    rt::CoreRelease release;
    const SOCKET s = accept(listener, reinterpret_cast<sockaddr*>(&a), &len);
    if (s == INVALID_SOCKET) return Ret(c, (u64)Wsa(SOCKET_ERROR));
    ToGuestAddr(a, len, Ptr<u8>(c, 1), Ptr<u32>(c, 2));
    Ret(c, (u64)(s64)Add({s, nullptr}));
}
NATIVE("nnsocketGetSockName") {
    sockaddr_storage a;
    int len = sizeof a;
    const s64 r = Wsa(getsockname(Sock(c), reinterpret_cast<sockaddr*>(&a), &len));
    if (r == 0) ToGuestAddr(a, len, Ptr<u8>(c, 1), Ptr<u32>(c, 2));
    Ret(c, (u64)r);
}
NATIVE("nnsocketGetPeerName") {
    sockaddr_storage a;
    int len = sizeof a;
    const s64 r = Wsa(getpeername(Sock(c), reinterpret_cast<sockaddr*>(&a), &len));
    if (r == 0) ToGuestAddr(a, len, Ptr<u8>(c, 1), Ptr<u32>(c, 2));
    Ret(c, (u64)r);
}
NATIVE("nnsocketSend") {
    const SOCKET s = Sock(c);
    rt::CoreRelease release;
    Ret(c, (u64)Wsa(send(s, Ptr<const char>(c, 1), (int)c->x[2], HostMsgFlags((int)c->x[3]))));
}
NATIVE("nnsocketRecv") {
    const SOCKET s = Sock(c);
    rt::CoreRelease release;
    Ret(c, (u64)Wsa(recv(s, Ptr<char>(c, 1), (int)c->x[2], HostMsgFlags((int)c->x[3]))));
}
// msghdr: name, namelen (u32), iov {void*, size_t}*, iovlen (u32), ... Sent/received as one buffer.
static s64 MsgCall(Cpu* c, bool send_msg) {
    const u8* g = Ptr<u8>(c, 1);
    u64 name, iov;
    u32 namelen, iovlen;
    std::memcpy(&name, g, 8);
    std::memcpy(&namelen, g + 8, 4);
    std::memcpy(&iov, g + 16, 8);
    std::memcpy(&iovlen, g + 24, 4);
    struct GuestIov { u64 base, len; };
    const auto* v = reinterpret_cast<const GuestIov*>(iov);
    size_t total = 0;
    for (u32 i = 0; i < iovlen; ++i) total += v[i].len;
    std::vector<char> buffer(total);
    const SOCKET s = Sock(c);
    if (send_msg) {
        size_t at = 0;
        for (u32 i = 0; i < iovlen; ++i) { std::memcpy(buffer.data() + at, reinterpret_cast<void*>(v[i].base), v[i].len); at += v[i].len; }
        sockaddr_storage a;
        if (name) {
            const int len = ToHostAddr(reinterpret_cast<const u8*>(name), namelen, a);
            return Wsa(sendto(s, buffer.data(), (int)total, HostMsgFlags((int)c->x[2]), reinterpret_cast<sockaddr*>(&a), len));
        }
        return Wsa(send(s, buffer.data(), (int)total, HostMsgFlags((int)c->x[2])));
    }
    const s64 r = Wsa(recv(s, buffer.data(), (int)total, HostMsgFlags((int)c->x[2])));
    for (size_t i = 0, at = 0; r > 0 && i < iovlen && at < (size_t)r; ++i) {
        const size_t n = std::min<size_t>(v[i].len, (size_t)r - at);
        std::memcpy(reinterpret_cast<void*>(v[i].base), buffer.data() + at, n);
        at += n;
    }
    return r;
}
NATIVE("nnsocketSendMsg") { rt::CoreRelease release; Ret(c, (u64)MsgCall(c, true)); }
NATIVE("nnsocketRecvMsg") { rt::CoreRelease release; Ret(c, (u64)MsgCall(c, false)); }
NATIVE("nnsocketPoll") { rt::CoreRelease release; Ret(c, (u64)Poll(Ptr<GuestPollFd>(c, 0), c->x[1], (int)c->x[2])); }
NATIVE("nnsocketFcntl") {
    Fd f;
    if (!Get((s32)c->x[0], f)) return Ret(c, (u64)Fail(9));
    const int cmd = (int)c->x[1];
    if (cmd == 3) { // F_GETFL: O_RDWR, plus O_NONBLOCK for non-blocking eventfds (sockets: not queryable)
        Ret(c, (u64)(2 | (f.event && f.event->nonblocking ? 4 : 0)));
    } else if (cmd == 4) { // F_SETFL
        const bool nonblocking = c->x[2] & 4;
        if (f.event) { std::lock_guard lock{fds_mutex}; f.event->nonblocking = nonblocking; return Ret(c, 0); }
        u_long on = nonblocking;
        Ret(c, (u64)Wsa(ioctlsocket(f.socket, FIONBIO, &on)));
    } else {
        Ret(c, 0);
    }
}
static s64 SetOpt(Cpu* c) {
    if (!HostOption((int)c->x[1], (int)c->x[2])) return 0;
    return Wsa(setsockopt(Sock(c), (int)c->x[1], (int)c->x[2], Ptr<const char>(c, 3), (int)c->x[4]));
}
NATIVE("nnsocketSetSockOpt") { Ret(c, (u64)SetOpt(c)); }
NATIVE("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj") { Ret(c, (u64)SetOpt(c)); }
NATIVE("nnsocketGetSockOpt") {
    if (!HostOption((int)c->x[1], (int)c->x[2])) {
        std::memset(Ptr(c, 3), 0, *Ptr<u32>(c, 4));
        return Ret(c, 0);
    }
    int len = (int)*Ptr<u32>(c, 4);
    const s64 r = Wsa(getsockopt(Sock(c), (int)c->x[1], (int)c->x[2], Ptr<char>(c, 3), &len));
    if (r == 0 && (int)c->x[1] == 0xffff && (int)c->x[2] == 0x1007) {
        s32& e = *Ptr<s32>(c, 3);
        if (e >= 10000 && e < 11000) e -= 10000;
    }
    *Ptr<u32>(c, 4) = (u32)len;
    Ret(c, (u64)r);
}
NATIVE("nnsocketGetLastErrno") { Ret(c, (u64)(s64)last_errno); }
NATIVE("nnsocketSetLastErrno") { last_errno = (s32)c->x[0]; }
NATIVE("nnsocketInetHtonl") { Ret(c, htonl((u32)c->x[0])); }
NATIVE("nnsocketInetNtohl") { Ret(c, ntohl((u32)c->x[0])); }
NATIVE("nnsocketInetHtons") { Ret(c, htons((u16)c->x[0])); }
NATIVE("nnsocketInetNtohs") { Ret(c, ntohs((u16)c->x[0])); }
// Offline: no host names resolve (EAI_NONAME).
NATIVE("nnsocketGetAddrInfo") { *Ptr<u64>(c, 3) = 0; Ret(c, 8); }
NATIVE("nnsocketGetAddrInfoWithOptions") { *Ptr<u64>(c, 3) = 0; Ret(c, 8); }
NATIVE("nnsocketFreeAddrInfo") {}
NATIVE("nnsocketGAIStrError") {
    static const char text[] = "host name lookup is unavailable offline";
    Ret(c, reinterpret_cast<u64>(text));
}
NATIVE("nnsocketResolverGetOption") { Ret(c, 0); }
NATIVE("nnsocketResolverSetOption") { Ret(c, 0); }

// eventfd: flags 1 = semaphore, 4 = non-blocking.
NATIVE("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE") {
    auto e = std::make_shared<EventFd>();
    e->counter = (u32)c->x[0];
    e->semaphore = c->x[1] & 1;
    e->nonblocking = c->x[1] & 4;
    Ret(c, (u64)(s64)Add({INVALID_SOCKET, e}));
}
NATIVE("_ZN2nn6socket11EventFdReadEiPm") {
    Fd f;
    if (!Get((s32)c->x[0], f) || !f.event) return Ret(c, (u64)Fail(9));
    rt::CoreRelease release;
    std::unique_lock lock{fds_mutex};
    while (!f.event->counter) {
        if (f.event->nonblocking) return Ret(c, (u64)Fail(35)); // EAGAIN
        events_changed.wait(lock);
    }
    const u64 value = f.event->semaphore ? 1 : f.event->counter;
    f.event->counter -= value;
    *Ptr<u64>(c, 1) = value;
    events_changed.notify_all();
    Ret(c, 0);
}
NATIVE("_ZN2nn6socket12EventFdWriteEim") {
    Fd f;
    if (!Get((s32)c->x[0], f) || !f.event) return Ret(c, (u64)Fail(9));
    if (c->x[1] == ~0ULL) return Ret(c, (u64)Fail(22));
    std::lock_guard lock{fds_mutex};
    f.event->counter += c->x[1];
    events_changed.notify_all();
    Ret(c, 0);
}

#endif // _WIN32
