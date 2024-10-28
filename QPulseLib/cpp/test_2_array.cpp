#include <iostream>

using namespace std;

int main() {
    int a[2][2] = {1, 2, 3, 4};

    int *p = (int*)a;
    cout << *(p + 3) << endl;
    return 0;
}