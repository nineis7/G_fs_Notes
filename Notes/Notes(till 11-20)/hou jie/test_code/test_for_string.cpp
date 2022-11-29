#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>
using namespace std;

class String
{
private:
    /* data */
    char* m_data; //只存放指针，内存动态分配
public:
    String(const char* cstr = 0);
    //The Big Three
    String(const String& str);
    String& operator = (const String& str);
    ~String();

    char* get_c_str() const {return m_data;}

    //ostream& operator << (ostream& os, const String& str);
};

inline
String::String(const char* cstr)
{
    if(cstr){
        m_data = new char[strlen(cstr) + 1];
        strcpy(m_data, cstr);
    }
    else{
        m_data = new char[1];
        *m_data = '\0';
    }
}

//不加引用会报错：
//类 "String" 的复制构造函数不能带有 "String" 类型的参数
inline
String::String(const String& str)
{
    m_data = new char[strlen(str.m_data) + 1];
    strcpy(m_data, str.m_data);
}

inline String&
String::operator = (const String& str)
{
    //自我检测
    if(this == &str) return *this;

    delete[] m_data;
    m_data = new char[strlen(str.m_data) + 1];
    strcpy(m_data, str.m_data);
    return *this;
}

inline ostream&
operator << (ostream& os, const String& str)
{
    return os << str.get_c_str();
}

String::~String()
{
    delete[] m_data;
}


int main(){
    String s1();
    String s2("hello");
    String s3(s2);
    s3 = s2;
    //s1 = s2;
    cout << s3 << endl;
    cout << s1 << endl;
    return 0;
}