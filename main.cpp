//
// Created by Varun Thvar on 1/15/24.
//

#include "main.h"
#include "iostream"
using namespace std;

int main()
{
   int* numbers = new int[3];
   *(numbers+4) = 4;

   for (int i = 0; i < 5; i++)
   {
      cout << numbers[i] << endl;
   }

   cout << "Hello World!";
}