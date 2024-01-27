/*
 * Author: Varun Thvar
 * Date of Creation: 26 January 2023
 * Description: This is an implementation of a simple A-B-1 Network designed for simple boolean algebra problems.
 */

#include <cmath>
#include <iostream>
#include <time.h>

#define MAX_ITERATIONS (int) 100000
#define LAMBDA (double) 0.3
#define RANDOM_MIN (double) -1.5
#define RANDOM_MAX (double) 1.5
#define ERROR_THRESHOLD (double) (2.0 * pow(10, -4))



using namespace std;

/**
 * This implements the sigmoid function, which is defined by
 *    sigmoid(x) = 1/(1 + e^(-x))
 */
double sigmoid(double value)
{
    return 1/(1 + exp(-1 * value));
}

/**
 * This implements the derivative of the sigmoid function, which is defined by
 *    (d/dx) sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)), where sigmoid is defined by
 *    sigmoid(x) = 1/(1 + e^(-x))
 */
double sigmoidPrime(double value)
{
    double sig = sigmoid(value);
    return sig * (1 - sig);
}

/**
 *
 */
void printArray(double *array, int size)
{
   for (int looper = 0; looper < size; looper++) cout << array[looper] << ", ";
   cout << endl << "Size of Array: " << size << endl;
   return;
}

/**
 * Accept a value representing seconds elapsed and print out a decimal value in easier to digest units
 * The magic numbers 1000., 60., 60., 24. and 7. need to be replaced with constants.
 */
void printTime(double seconds)
{
   double minutes, hours, days, weeks;

   printf("Elapsed time: ");

   if (seconds < 1.)
      printf("%g milliseconds", seconds * 1000.);
   else if (seconds < 60.)
      printf("%g seconds", seconds);
   else
   {
      minutes = seconds / 60.;

      if (minutes < 60.)
         printf("%g minutes", minutes);
      else
      {
         hours = minutes / 60.;

         if (hours < 24.)
            printf("%g hours", hours);
         else
         {
            days = hours / 24.;

            if (days < 7.)
               printf("%g days", days);
            else
            {
               weeks = days / 7.;

               printf("%g weeks", weeks);
            }
         } // if (hours < 24.)...else
      } // if (minutes < 60.)...else
   } // else if (seconds < 60.)...else

   printf("\n\n");
   return;
} // void printTime(double seconds)

/**
 *
 */
struct NeuralNetwork
{
   int k;
   int j;
   int numHiddenLayers;
   int lambda;
   float errorThreshold;
   int maxIterations;
   double randMin;
   double randMax;

   double* a;
   double* h;
   double F0;
   double* weights0j;
   double** weightsjk;

   bool training;
   int I, J, K;
   double thetaj;
   double thetai;

   int training_time;
   int running_time;
   int dummyStart;
   int dummyEnd;
   int iterations;
   float error_reached;
   string reasonForEndOfTraining;

   /**
    * Outputs a random value based on the range as given in the configuration parameters
    */
   double randomValue()
   {
      return rand() * (randMin - randMax) + randMin;
   }

   /**
    * Sets the configuration parameters
    */
   void setConfigurationParameters(int numAct, int numHidLayer, int numHidInEachLayer, int lamb,
                                   float errorThres, int maxIter, double min, double max, bool train)
   {
      this->k = numAct;
      this->numHiddenLayers = numHidLayer;
      this->j = numHidInEachLayer;
      this->lambda = lamb;
      this->errorThreshold = errorThres;
      this->maxIterations = maxIter;
      this->randMin = min;
      this->randMax = max;
      this->training = train;
      return;
   } // void setConfigurationParameters(int numAct, int numHidLayer ...

  /**
   *
   */
   void echoConfigurationParameters()
   {
      cout << "Number of Activations: " << this->k << endl;
      cout << "Number of Hidden Nodes in Each Hidden Layer: " << this->j << endl;
      cout << "Number of Hidden Layers: " << this->numHiddenLayers << endl;
      cout << "Lambda Value: " << this->lambda << endl;
      cout << "Error Threshold: " << this->errorThreshold << endl;
      cout << "Maximum Number of Iterations: " << this->maxIterations << endl;
      cout << "Random Value Minimum: " << this->randMin << endl;
      cout << "Random Value Maximum: " << this->randMax << endl;
      return;
   } //void echoConfigurationParameters()

   /**
    *
    */
   void allocateArrayMemory()
   {
      a = new double[k];
      h = new double[j];

      weights0j = new double[j];

      weightsjk = new double*[j];
      for (J = 0; J < j; ++J) weightsjk[J] = new double[k];

      cout << "Allocated Memory!" << endl;
      return;
   } //void allocateArrayMemory()

   /**
    *
    */
   void populateArrays()
   {
      if (j == 2 && k == 2 && numHiddenLayers == 1)
      {
         weights0j[0] = 0.103;
         weights0j[1] = 0.23;

         weightsjk[0][0] = 1;
         weightsjk[1][0] = 1;
         weightsjk[0][1] = 1;
         weightsjk[1][1] = 1;
      } // if (numHiddensInEachLayer == 2 ...
      else
      {
         for (int J = 0; J < j; ++J) weights0j[J] = randomValue();

         for (int J = 0; J < j; ++J) for (int K = 0; K < k; ++K) weightsjk[J][K] = randomValue();
      }

      cout << "Populated Arrays!" << endl;

      return;
   } //void populateArrays()

   /**
    *
    */
   void Train()
   {
      error_reached = 0.0;

      return;
   }

   /**
    *
    */
   void Run(double *inputValues)
   {
      a = inputValues;

      for (J = 0; J < j; ++J)
      {
         thetaj = 0;
         for (K = 0; K < k; ++K)
         {
            thetaj += a[K] * weightsjk[J][K];
         }
         h[J] = sigmoid(thetaj);
      }

      thetai = 0;
      for (J = 0; J < j; ++J)
      {
         thetai += h[J] * weights0j[J];
      }
      F0 = sigmoid(thetai);

      cout << "Output: " << F0 << endl;
      return;
   } // void Run(double inputValues[])

   /**
    *
    */
   void reportResults()
   {
      if (training)
      {
         cout << "Reason for Termination: " << reasonForEndOfTraining << endl;

         cout << "Training Time Taken: ";
         printTime(training_time);
         cout << endl;

         cout << "Error Reached: " << error_reached << endl;
         cout << "Iterations reached: " << iterations << endl;
      } // if (training)

      else
      {
         cout << "Running Time Taken: ";
         printTime(running_time);
         cout << endl;

         cout << "Error Reached: " << error_reached << endl;
         cout << "Iterations reached: " << iterations << endl;
      }
      return;
   } // reportResults()
}; // struct NeuralNetwork

/**
 *
 */
int main(int argc, char *argv[])
{
   /**
    * Creating and Configurating the Network based on pre-determined constants and designs
    */
   auto* network = new NeuralNetwork();
   network->setConfigurationParameters(2, 1, 2, LAMBDA,
      ERROR_THRESHOLD, MAX_ITERATIONS, RANDOM_MIN, RANDOM_MAX, true);
   network->echoConfigurationParameters();

   network->allocateArrayMemory(); // Allocating Arrays in Network
   network->populateArrays();

   // network->Train(); // Training the Network using predetermined training data
   // network->reportResults();

   network->training = false; // Running the Network using test data
   double testdata[] = {0.0, 1.0};
   network->Run(testdata);
   // network->reportResults();

   return 0;
}
