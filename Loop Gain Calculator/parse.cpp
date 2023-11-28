#include <fstream>
#include <regex>

#include "parse.hpp"

lt::response *parse(filesystem::path path)
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
    wsmatch headerMatches;
    wregex headerRegex(LR"("
            "Title:.+\n"
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
            "((?:.+\n)+)Binary:)");

    if (!regex_match(header, headerMatches, headerRegex))
        return NULL;

    file.read(NULL, 3);

    size_t numVars = stoi(headerMatches[2]);

    wstring flags(headerMatches[1]);

    lt::response *res = (lt::response *)malloc(sizeof(lt::response));
    res->numPoints = stoi(headerMatches[3]);
    res->freq = (double *)malloc(sizeof(double) * res->numPoints);
    res->response = map<wstring, float **>();

    wstring varString(headerMatches[4]);
    wregex varRegex(LR"(\t\d+\t((?:V|I|Ix)\((?:\w|:)+\))\t\w+)");

    wsregex_iterator varBegin(varString.begin(), varString.end(), varRegex);
    wsregex_iterator varEnd;

    streampos pos = file.tellg();
    file.seekg(0, ios::end);
    res->numCases = (file.tellg() - pos) / (sizeof(double) + numVars * sizeof(float)) / res->numPoints;
    file.seekg(pos);

    for (wsregex_iterator i = varBegin; i != varEnd; ++i)
    {
        wstring var = (*i).str();
        float **varArray = (float **)malloc(sizeof(float *) * res->numCases);
        for (size_t j = 0; j < res->numCases; j++)
        {
            varArray[j] = (float *)malloc(sizeof(float) * res->numPoints);
        }
        res->response.insert(pair<wstring, float **>(var, varArray));
    }

    for (size_t i = 0; i < res->numCases; i++)
    {
        for (size_t j = 0; j < res->numPoints; j++)
        {
            file.read((wchar_t *)&res->freq[i], sizeof(double));

            for (map<wstring, float **>::iterator it = res->response.begin(); it != res->response.end(); ++it)
            {
                file.read((wchar_t *)&it->second[i][j], sizeof(float));
            }
        }
    }

    return res;
}
