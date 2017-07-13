#include <iostream>
#include <bits/stdc++.h>

using namespace std;

int main(){
    string input = "\"The Canterbury Tales\" by Chaucer, G.";
    sscanf(input.c_str(), "\"%[^\"]\" by %[^\n\r]", t, a);    
    return 0;
}