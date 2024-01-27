/*
 * Author: Varun Thvar
 * Date of Creation: 26 January 2023
 * Description: ()
 */

#include <iostream>
#define MAX_ITERATIONS (int) 100000
#define LAMBDA (double) 0.3
#define RANDOM_MIN (double) -1.5
#define RANDOM_MAX (double) 1.5
#define ERROR_THRESHOLD (double) (2.0 * pow(10, -4))



using namespace std;

/**
 *
 */
double sigmoid(double value)
{
    return 1/(1 + exp(-1 * value));
}

/**
 * \brief
 * \param value
 * \return
 */
double sigmoidPrime(double value)
{
    double sig = sigmoid(value);
    return sig * (1 - sig);
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
   int numActivations;
   int numHiddensInEachLayer;
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
   int i, j, k;
   int I, J, K;

   int training_time;
   int running_time;
   int iterations;
   float error_reached;
   string reasonForEndOfTraining;

   /**
    *
    */
   double randomValue()
   {
      return rand() * (randMin - randMax) + randMin;
   }

   /**
    *
    */
   void setConfigurationParameters(int numAct, int numHidLayer, int numHidInEachLayer, int lamb,
                                   float errorThres, int maxIter, double min, double max, bool train)
   {
      this->numActivations = numAct;
      this->numHiddensInEachLayer = numHidLayer;
      this->numHiddenLayers = numHidInEachLayer;
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
      cout << "Number of Activations: " << this->numActivations << endl;
      cout << "Number of Hidden Nodes in Each Hidden Layer: " << this->numActivations << endl;
      cout << "Number of Hidden Layers: " << this->numActivations << endl;
      cout << "Lambda Value: " << this->numActivations << endl;
      cout << "Error Threshold: " << this->numActivations << endl;
      cout << "Maximum Number of Iterations: " << this->numActivations << endl;
      cout << "Random Value Minimum: " << this->numActivations << endl;
      cout << "Random Value Maximum: " << this->numActivations << endl;
      return;
   } //void echoConfigurationParameters()

   /**
    *
    */
   void allocateArrayMemory()
   {
      a = new double[numActivations];
      h = new double[numHiddensInEachLayer];

      weights0j = new double[numHiddensInEachLayer];

      weightsjk[numHiddensInEachLayer] = new double[numHiddensInEachLayer];
      for (J = 0; J < numHiddensInEachLayer; ++J) this->weightsjk[J] = new double[numActivations];
      return;
   } //void allocateArrayMemory()

   /**
    *
    */
   void populateArrays()
   {
      if (numHiddensInEachLayer == 2 && numActivations == 2)
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
         for (int J = 0; J < numHiddensInEachLayer; ++J) weights0j[J] = randomValue();

         for (int J = 0; J < numHiddensInEachLayer; ++J) for (int K = 0; K < numActivations; ++K) weightsjk[J][K] = randomValue();
      }
      return;
   } //void populateArrays()

   /**
    *
    */
   void Train()
   {

   }

   /**
    *
    */
   void Run()
   {

   }

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

   } // reportResults()
}; // struct NeuralNetwork

/**
 *
 */
int main2(int argc, char *argv[])
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

   network->Train(); // Training the Network using predetermined training data
   network->reportResults();

   network->training = false; // Running the Network using test data
   network->Run();
   network->reportResults();

   return 0;
}
