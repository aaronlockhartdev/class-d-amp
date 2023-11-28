#include <map>
#include <string>
#include <filesystem>

using namespace std;

namespace lt
{
    struct response
    {
        size_t numPoints;
        size_t numCases;

        double *freq;
        map<wstring, float **> response;
    };
}

lt::response *parse(filesystem::path path);