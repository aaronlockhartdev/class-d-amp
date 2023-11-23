#include <fstream>

#include "ltspice-parse.hpp"

ltResponse parse(string path) {
    ifstream file;
    file.open(path, ios::in | ios::binary);
    
}