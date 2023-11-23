#include <map>
#include <string>

using namespace std;

struct ltResponse {
    size_t numPoints;

    double* frq;
    map<string, float*> response;
};

ltResponse parse(string path);
