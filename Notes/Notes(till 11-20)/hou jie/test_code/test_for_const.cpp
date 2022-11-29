#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
using namespace std;

class complex
{
private:
    /* data */
    double re;
    double im;
public:
    complex(double r = 0, double i = 0)
    : re(r), im(i) {}
    ~complex() {}

    double real() const {return re;}
    double imag() const {return im;}
    //ostream& operator << (ostream&, const complex&);

    //友元函数测试无误
    friend ostream& operator << (ostream&, const complex&);
};


inline ostream&
operator << (ostream& os, const complex& r)
{
    return os << "(" << r.re << ", "
                << r.im << ")" << endl;
}

/*
inline ostream&
complex::operator << (ostream& os, const complex& r)
{
    return os << "(" << r.re << ", "
                << r.im << ")" << endl;
}
*/

inline double
real(const complex& r)
{
    return r.real();
}

int main(){
    complex c1(2.5, 3.4);

    cout << real(c1) << endl;
    cout << c1 << endl;
    return 0;
}