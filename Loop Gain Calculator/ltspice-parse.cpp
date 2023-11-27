#include <filesystem>
#include <fstream>
#include <regex>

#include "ltspice-parse.hpp"

ltResponse *parse(filesystem::path path)
{
    wifstream file(path, ios::in | ios::binary);

    wstring endHeader = L"Binary:\n"s;

    vector<wchar_t> headerVec;

    while (true)
    {
        wchar_t c = file.get();

        if (c == EOF)
            return NULL;

        headerVec.push_back(c);

        if (
            headerVec.size() >= endHeader.length() &&
            wstring(headerVec.end() - endHeader.length(), headerVec.end()) == endHeader)
            break;
    }

    wstring header(headerVec.begin(), headerVec.end());

    wregex regex(LR"(Title:.+\n"
        "Date:.+\n"
        "Plotname: AC Analysis\n"
        "Flags:((?: \w+)*)\n"
        "No. Variables: (\d+)\n"
        "No. Points:\h+(\d+)\n"
        "Offset:\h+(?:\d|\.|e|\+)+\n"
        "Command: .+\n"
        "(?:Backannotation: .+\n)*"
        "Variables:\n"
        "\t0\tfrequency\tfrequency\n"
        "(?:(?:\t(\d+)\t(?:V|I|Ix)\(fb\)\tvoltage\n)|(?:.+\n))+"
        "Binary:)");

    wsmatch sm;

    if (!regex_match(header, sm, regex))
        return NULL;

    ltResponse *res = (ltResponse *)malloc(sizeof(ltResponse));

    size_t numVars = stoi(sm[2]);
    size_t targetIdx = stoi(sm[4]);

    wstring flags(sm[1]);

    res->numPoints = stoi(sm[3]);

    file.read(NULL, 3);

    for (size_t i = 0; i < res->numPoints; i++)
    {
    }
}