#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>

#include "ltspice-parse.hpp"

ltResponse *parse(filesystem::path path)
{
    wifstream file(path, ios::in | ios::binary);

    u16string endHeader = u"Binary:\n"s;

    vector<wchar_t> headerVec;

    while (true)
    {
        wchar_t c = file.get();

        if (c == EOF)
            return NULL;

        headerVec.push_back(c);

        if (
            headerVec.size() >= endHeader.length() &&
            u16string(headerVec.end() - endHeader.length(), headerVec.end()) == endHeader)
            break;
    }

    u16string header(headerVec.begin(), headerVec.end());

    string regex =
        ""
        "";
}