#include "pttf.hpp"

int main(int argc, char const *argv[])
{
    Point(mp1, p1, 1, 2, 3);
    Point(mc1, c1, 0, 0, 0);
    Point(mc2, c2, 3, 3, 3);
    pttf::add_xyz("O", c1);
    pttf::add_xyz("A", c2);
    np::Numcpp<double> p2 = pttf::transform(p1, "O", "A", 60, 60, 60);
    std::cout << p2 << std::endl;
    return 0;
}
