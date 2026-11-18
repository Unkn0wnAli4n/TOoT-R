// The game's log on disk. Everything the runtime says goes to stderr; a player who starts the game from a shortcut
// never sees that, and a start that failed left nothing to send. Each run now writes <data>/logs/tootr.log (the three
// runs before it are kept as tootr.1.log .. tootr.3.log), and a fatal error says so in a message box.
#pragma once
#include <string>

namespace rt {
void OpenLogFile();                    // first thing in main: stderr and stdout go to the file (and on to the console)
const std::string& LogFilePath();      // "" when no file is kept (OOT_NO_LOG_FILE=1, or the folder cannot be written)
void ShowFatal(const char* message);   // a message box with the error and the log's place; nothing with OOT_NO_DIALOG=1
} // namespace rt
