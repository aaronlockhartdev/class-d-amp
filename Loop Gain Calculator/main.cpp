/*----------------------
|  Main function
-----------------------*/

#include <stdio.h>

#include "parse.hpp"
#include "compile.hpp"

int main(int argc, char *argv[])
{
    lt::response *res = parse();
}