/*
authored by Joseph Augustus Fiume
*/

#include <dirent.h>

int loadDirectory(vector<string>& outFiles, string path,
                  const char* filePath, const char* fileType)
{
    vector<string> inFiles;
    // get files from directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(filePath)) != NULL)
    {
      while ((ent = readdir (dir)) != NULL)
      {
        if (strstr(ent->d_name, fileType))
            inFiles.push_back(ent->d_name);
      }
      closedir (dir);
    }
    else
    {
      // could not open directory
      perror ("");
      return EXIT_FAILURE;
    }
    // add the file path to the file names
    for (int i = 0; i < inFiles.size(); ++i)
    {
        inFiles[i] = path + "/" + inFiles[i];
    }
    // create a vector of file names
    for (int i = 0; i < inFiles.size(); ++i)
    {
        outFiles.push_back(inFiles[i]);
    }
    return 0;
}

