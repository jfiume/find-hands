/*
authored by Joseph Augustus Fiume
*/

/*
 Allows for loading of a directory.
*/

#ifndef LOADDIRECTORY_H
#define LOADDIRECTORY_H


int loadDirectory(std::vector<std::string>& outFiles, std::string path,
                  const char* filePath, const char* fileType);

#include "../src/LoadDirectory.cpp"
#endif //LOADDIRECTORY_H
