// nn::socket on host sockets. The console's socket API is BSD (FreeBSD constants and sockaddr with
// a length byte); values are translated to Linux. The game stays offline: name resolution always
// fails, so only loopback and the library's internal plumbing (eventfds, local sockets) work.

#ifndef _WIN32 // (Windows: socket_win32.cpp)
#include <cerrno>
#include <cstring>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../native.h"

using namespace rt;

namespace {

thread_local s32 last_errno = 0;

s32 BsdErrno(int e) {
    switch (e) {
    case EAGAIN: return 35;
    case EINPROGRESS: return 36;
    case EALREADY: return 37;
    case ENOTSOCK: return 38;
    case EDESTADDRREQ: return 39;
    case EMSGSIZE: return 40;
    case EPROTOTYPE: return 41;
    case ENOPROTOOPT: return 42;
    case EPROTONOSUPPORT: return 43;
    case EOPNOTSUPP: return 45;
    case EAFNOSUPPORT: return 47;
    case EADDRINUSE: return 48;
    case EADDRNOTAVAIL: return 49;
    case ENETDOWN: return 50;
    case ENETUNREACH: return 51;
    case ECONNABORTED: return 53;
    case ECONNRESET: return 54;
    case ENOBUFS: return 55;
    case EISCONN: return 56;
    case ENOTCONN: return 57;
    case ETIMEDOUT: return 60;
    case ECONNREFUSED: return 61;
    case EHOSTUNREACH: return 65;
    default: return e; // EINTR, EBADF, EINVAL, EPIPE, EFAULT... share values
    }
}

s64 Result(s64 r) {
    if (r < 0) last_errno = BsdErrno(errno);
    return r;
}

// BSD sockaddr {u8 len; u8 family; data...} <-> Linux sockaddr {u16 family; data...}.
socklen_t ToHostAddr(const u8* guest, socklen_t len, sockaddr_storage& host) {
    std::memset(&host, 0, sizeof host);
    if (!guest || len < 2) return 0;
    const u8 family = guest[1];
    host.ss_family = family == 28 ? AF_INET6 : family; // BSD AF_INET6 = 28
    std::memcpy(reinterpret_cast<u8*>(&host) + 2, guest + 2, std::min<size_t>(len - 2, sizeof host - 2));
    return family == 28 ? sizeof(sockaddr_in6) : len;
}

void ToGuestAddr(const sockaddr_storage& host, socklen_t host_len, u8* guest, u32* guest_len) {
    if (!guest || !guest_len) return;
    const u32 n = std::min<u32>(*guest_len, (u32)host_len);
    std::memcpy(guest, &host, n);
    if (n >= 2) {
        guest[0] = (u8)host_len;
        guest[1] = host.ss_family == AF_INET6 ? 28 : (u8)host.ss_family;
    }
    *guest_len = host_len;
}

int HostType(int type) {
    int host = type & 0xffff;
    if (type & 0x20000000) host |= SOCK_NONBLOCK;
    if (type & 0x10000000) host |= SOCK_CLOEXEC;
    return host;
}

int HostMsgFlags(int f) {
    int h = f & (MSG_OOB | MSG_PEEK | MSG_DONTROUTE | MSG_WAITALL);
    if (f & 0x80) h |= MSG_DONTWAIT;
    return h | MSG_NOSIGNAL;
}

// Socket option translation; returns false for options without a host equivalent (ignored).
bool HostOption(int level, int opt, int& hl, int& ho) {
    if (level == 0xffff) {
        hl = SOL_SOCKET;
        switch (opt) {
        case 0x4: ho = SO_REUSEADDR; return true;
        case 0x8: ho = SO_KEEPALIVE; return true;
        case 0x20: ho = SO_BROADCAST; return true;
        case 0x80: ho = SO_LINGER; return true;
        case 0x200: ho = SO_REUSEPORT; return true;
        case 0x1001: ho = SO_SNDBUF; return true;
        case 0x1002: ho = SO_RCVBUF; return true;
        case 0x1005: ho = SO_SNDTIMEO; return true;
        case 0x1006: ho = SO_RCVTIMEO; return true;
        case 0x1007: ho = SO_ERROR; return true;
        case 0x1008: ho = SO_TYPE; return true;
        default: return false;
        }
    }
    hl = level;
    ho = opt;
    if (level == IPPROTO_IPV6 && opt == 27) ho = IPV6_V6ONLY;
    return level == IPPROTO_TCP || level == IPPROTO_IP || level == IPPROTO_IPV6;
}

s64 SetOpt(Cpu* c) {
    int hl, ho;
    if (!HostOption((int)c->x[1], (int)c->x[2], hl, ho)) return 0;
    return Result(setsockopt((int)c->x[0], hl, ho, Ptr(c, 3), (socklen_t)c->x[4]));
}

} // namespace

NATIVE("nnsocketSocket") { Ret(c, (u64)Result(socket((int)c->x[0] == 28 ? AF_INET6 : (int)c->x[0], HostType((int)c->x[1]), (int)c->x[2]))); }
NATIVE("nnsocketClose") { Ret(c, (u64)Result(close((int)c->x[0]))); }
NATIVE("nnsocketShutdown") { Ret(c, (u64)Result(shutdown((int)c->x[0], (int)c->x[1]))); }
NATIVE("nnsocketListen") { Ret(c, (u64)Result(listen((int)c->x[0], (int)c->x[1]))); }
NATIVE("nnsocketBind") {
    sockaddr_storage a;
    const socklen_t len = ToHostAddr(Ptr<u8>(c, 1), (socklen_t)c->x[2], a);
    Ret(c, (u64)Result(bind((int)c->x[0], reinterpret_cast<sockaddr*>(&a), len)));
}
NATIVE("nnsocketConnect") {
    sockaddr_storage a;
    const socklen_t len = ToHostAddr(Ptr<u8>(c, 1), (socklen_t)c->x[2], a);
    rt::CoreRelease release;
    Ret(c, (u64)Result(connect((int)c->x[0], reinterpret_cast<sockaddr*>(&a), len)));
}
NATIVE("nnsocketAccept") {
    sockaddr_storage a;
    socklen_t len = sizeof a;
    rt::CoreRelease release;
    const int fd = (int)Result(accept((int)c->x[0], reinterpret_cast<sockaddr*>(&a), &len));
    if (fd >= 0) ToGuestAddr(a, len, Ptr<u8>(c, 1), Ptr<u32>(c, 2));
    Ret(c, (u64)fd);
}
NATIVE("nnsocketGetSockName") {
    sockaddr_storage a;
    socklen_t len = sizeof a;
    const s64 r = Result(getsockname((int)c->x[0], reinterpret_cast<sockaddr*>(&a), &len));
    if (r == 0) ToGuestAddr(a, len, Ptr<u8>(c, 1), Ptr<u32>(c, 2));
    Ret(c, (u64)r);
}
NATIVE("nnsocketGetPeerName") {
    sockaddr_storage a;
    socklen_t len = sizeof a;
    const s64 r = Result(getpeername((int)c->x[0], reinterpret_cast<sockaddr*>(&a), &len));
    if (r == 0) ToGuestAddr(a, len, Ptr<u8>(c, 1), Ptr<u32>(c, 2));
    Ret(c, (u64)r);
}
NATIVE("nnsocketSend") { rt::CoreRelease release; Ret(c, (u64)Result(send((int)c->x[0], Ptr(c, 1), c->x[2], HostMsgFlags((int)c->x[3])))); }
NATIVE("nnsocketRecv") { rt::CoreRelease release; Ret(c, (u64)Result(recv((int)c->x[0], Ptr(c, 1), c->x[2], HostMsgFlags((int)c->x[3]) & ~MSG_NOSIGNAL))); }
// msghdr (both LP64): name, namelen (u32), iov, iovlen, control, controllen, flags; BSD uses u32
// counts where Linux uses size_t, so the header is rebuilt.
static s64 MsgCall(Cpu* c, bool send_msg) {
    const u8* g = Ptr<u8>(c, 1);
    msghdr m{};
    u64 name, iov, control;
    u32 namelen, iovlen, controllen;
    std::memcpy(&name, g, 8);
    std::memcpy(&namelen, g + 8, 4);
    std::memcpy(&iov, g + 16, 8);
    std::memcpy(&iovlen, g + 24, 4);
    std::memcpy(&control, g + 32, 8);
    std::memcpy(&controllen, g + 40, 4);
    sockaddr_storage a;
    if (name && send_msg) {
        m.msg_namelen = ToHostAddr(reinterpret_cast<const u8*>(name), namelen, a);
        m.msg_name = &a;
    }
    m.msg_iov = reinterpret_cast<iovec*>(iov); // iovec {void*, size_t} matches
    m.msg_iovlen = iovlen;
    const s64 r = send_msg ? Result(sendmsg((int)c->x[0], &m, HostMsgFlags((int)c->x[2])))
                           : Result(recvmsg((int)c->x[0], &m, HostMsgFlags((int)c->x[2]) & ~MSG_NOSIGNAL));
    (void)control;
    (void)controllen;
    return r;
}
NATIVE("nnsocketSendMsg") { rt::CoreRelease release; Ret(c, (u64)MsgCall(c, true)); }
NATIVE("nnsocketRecvMsg") { rt::CoreRelease release; Ret(c, (u64)MsgCall(c, false)); }
NATIVE("nnsocketPoll") { rt::CoreRelease release; Ret(c, (u64)Result(poll(Ptr<pollfd>(c, 0), c->x[1], (int)c->x[2]))); }
NATIVE("nnsocketFcntl") {
    const int fd = (int)c->x[0], cmd = (int)c->x[1];
    if (cmd == 3) { // F_GETFL
        const int f = (int)Result(fcntl(fd, F_GETFL));
        Ret(c, (u64)(f < 0 ? f : ((f & O_NONBLOCK) ? 4 : 0) | (f & 3)));
    } else if (cmd == 4) { // F_SETFL
        const int cur = fcntl(fd, F_GETFL);
        Ret(c, (u64)Result(fcntl(fd, F_SETFL, (c->x[2] & 4) ? (cur | O_NONBLOCK) : (cur & ~O_NONBLOCK))));
    } else {
        Ret(c, 0);
    }
}
NATIVE("nnsocketSetSockOpt") { Ret(c, (u64)SetOpt(c)); }
NATIVE("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj") { Ret(c, (u64)SetOpt(c)); }
NATIVE("nnsocketGetSockOpt") {
    int hl, ho;
    if (!HostOption((int)c->x[1], (int)c->x[2], hl, ho)) {
        std::memset(Ptr(c, 3), 0, *Ptr<u32>(c, 4));
        return Ret(c, 0);
    }
    socklen_t len = *Ptr<u32>(c, 4);
    const s64 r = Result(getsockopt((int)c->x[0], hl, ho, Ptr(c, 3), &len));
    if (r == 0 && hl == SOL_SOCKET && ho == SO_ERROR) *Ptr<s32>(c, 3) = BsdErrno(*Ptr<s32>(c, 3));
    *Ptr<u32>(c, 4) = len;
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

// eventfd: flags 1 = semaphore, 4 = non-blocking (BSD O_NONBLOCK).
NATIVE("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE") {
    int flags = EFD_CLOEXEC;
    if (c->x[1] & 1) flags |= EFD_SEMAPHORE;
    if (c->x[1] & 4) flags |= EFD_NONBLOCK;
    Ret(c, (u64)Result(eventfd((unsigned)c->x[0], flags)));
}
NATIVE("_ZN2nn6socket11EventFdReadEiPm") {
    const ssize_t n = read((int)c->x[0], Ptr(c, 1), 8);
    Ret(c, (u64)Result(n == 8 ? 0 : -1));
}
NATIVE("_ZN2nn6socket12EventFdWriteEim") {
    const u64 v = c->x[1];
    const ssize_t n = write((int)c->x[0], &v, 8);
    Ret(c, (u64)Result(n == 8 ? 0 : -1));
}

#endif // !_WIN32
