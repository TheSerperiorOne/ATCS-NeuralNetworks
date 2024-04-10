#include <iostream>

using namespace std;

int main1()
{
    std::cout << "Hello" << std::endl;
    return 0;

   //      weights = new double**[numLayers]; // Initializing weights
//      for (n = 1; n < numLayers; ++n)
//      {
//         weights[n] = new double*[LayerConfiguration[n]];
//         for (int k = 0; k < LayerConfiguration[n]; ++k)
//            weights[n][k] = new double[LayerConfiguration[n - 1]];
//      } // for (index = 0; index < numLayers; ++index)
}


/**
 * Prints array in a sequence for the purposes of reporting results after running. Follows the format
 *  value1, value2, value3, ... valueN. Specifically for arrays of ints.
 */

void printArray(int* arr, int length)
{
   for (int index = 0; index < length; ++index)
   {
      cout << arr[index];
      if (index != length - 1) cout << ", ";
   } // for (int index = 0; index < length; ++index)
   return;
} // void printArray(double* arr, int length)
