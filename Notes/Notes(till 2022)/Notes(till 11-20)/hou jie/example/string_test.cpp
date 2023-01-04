#include "string.h"
#include <iostream>

using namespace std;

int main()
{
  String s4();
  String s1("hello"); 
  String s2("world");
    
  String s3(s2);
  cout << s3 << endl;
  //s4 = s1;
  s3 = s1;
  cout << s3 << endl;     
  cout << s2 << endl;  
  cout << s1 << endl;      
}
