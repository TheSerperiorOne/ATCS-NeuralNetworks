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
#define MILLISECONDS_IN_SECOND (double) 1000.0
#define SECONDS_IN_MINUTE (double) 60.0
#define MINUTES_IN_HOUR (double) 60.0
#define HOURS_IN_DAY (double) 24.0
#define DAYS_IN_WEEK (double) 7.0



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
      printf("%g milliseconds", seconds * MILLISECONDS_IN_SECOND);
   else if (seconds < SECONDS_IN_MINUTE)
      printf("%g seconds", seconds);
   else
   {
      minutes = seconds / SECONDS_IN_MINUTE;

      if (minutes < MINUTES_IN_HOUR)
         printf("%g minutes", minutes);
      else
      {
         hours = minutes / MINUTES_IN_HOUR;

         if (hours < HOURS_IN_DAY)
            printf("%g hours", hours);
         else
         {
            days = hours / HOURS_IN_DAY;

            if (days < DAYS_IN_WEEK)
               printf("%g days", days);
            else
            {
               weeks = days / DAYS_IN_WEEK;

               printf("%g weeks", weeks);
            }
         } // if (hours < 24.)...else
      } // if (minutes < 60.)...else
   } // else if (seconds < 60.)...else

   printf("\n\n");
   return;
} // void printTime(double seconds)

/**
 * This is the structure of the Neural Network, which contains the following:
 *    k: Number of Activations
 *    j: Number of Hidden Nodes in each Hidden Layer
 *    numHiddenLayers: Number of Hidden Layers
 *    lambda: Lambda Value
 *    errorThreshold: Error Threshold
 *    maxIterations: Maximum number of iterations
 *    randMin: Minimum value of the random value assigned to weights
 *    randMax: Maximum value of the random value assigned to weights
 *    a: Activations
 *    h: Hidden Nodes
 *    F0: Output
 *    weights0j: Weights between the Input Layer and the Hidden Layers
 *    weightsjk: Weights between the Hidden Layers and the Output Layer
 *    training: Whether or not the network is in training mode (the alternative being running mode)
 *    I, J, K: Variables used for looping through the outputs, hidden layer, and activations respectively
 *    training_iterator: Used to iterate over all the test cases
 *    thetaj: Value used for calculating the Hidden Nodes
 *    thetai: Value used for calculating the Output
 *    training_time: Time taken for training the network
 *    running_time: Time taken for running the network
 *    dummyStart: Dummy variable used for timing
 *    dummyEnd: Dummy variable used for timing
 *    iterations: Number of iterations taken during training
 *    error_reached: Error value reached at the end of training or running
 *    reasonForEndOfTraining: Reason for the end of training
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
   int training_iterator;
   double thetaj;
   double thetai;

   double lowerOmega;
   double lowerPsi;
   double EPrimeJ0;
   double deltaWj0;
   double capitalOmega;
   double capitalPsi;
   double EPrimeJK;
   double deltaWJK; // TODO: Add to javadoc above

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
   } // double randomValue()

   /**
    * Sets the configuration parameters for the network based on the following parameters:
    *   numAct: Number of Activations
    *   numHidLayer: Number of Hidden Layers
    *   numHidInEachLayer: Number of Hidden Nodes in each Hidden Layer
    *   lamb: Lambda Value
    *   errorThres: Error Threshold
    *   maxIter: Maximum number of iterations
    *   min: Minimum value of the random value assigned to weights
    *   max: Maximum value of the random value assigned to weights
    *   train: Whether or not the network is in training mode (the alternative being running mode)
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
   * Outputs the configuration parameters for the network
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
    * Allocates memory for the activations, hidden nodes, and weights of the network
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
    * Populates the arrays with random values, unless the network is a 2-2-1 network, in which it manually overrides
    *    the values to match a set of pre-determined values
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
      } // else

      cout << "Populated Arrays!" << endl;

      return;
   } //void populateArrays()

   /**
    * Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
    *    and the Random Number Range. To be used before training and/or running.
    */
   void checkNetwork()
   {
      cout << "Network Type: " << k << "-" << j << "-" << 1 << endl;
      cout << "Lambda Value: " << this->lambda << endl;
      cout << "Error Threshold: " << this->errorThreshold << endl;
      cout << "Maximum Number of Iterations: " << this->maxIterations << endl;
      cout << "Random Value Minimum: " << this->randMin << endl;
      cout << "Random Value Maximum: " << this->randMax << endl;

      return;
   }

   /**
    * Trains the network using predetermined training data
    */
   void Train(double** data, double* answers) // TODO: Figure out how to train network
   {
      checkNetwork();
      error_reached = 0.0;

      for (training_iterator = 0; training_iterator < maxIterations && error_reached > errorThreshold; ++training_iterator)
      {
         checkNetwork();
         a = data[training_iterator % 4];

         for (J = 0; J < j; ++J)
         {
            thetaj = 0;
            for (K = 0; K < k; ++K)
            {
               thetaj += a[K] * weightsjk[J][K];
            } // for (K = 0; K < k; ++K)
            h[J] = sigmoid(thetaj);
         } // for (J = 0; J < j; ++J)

         thetai = 0;
         for (J = 0; J < j; ++J)
         {
            thetai += h[J] * weights0j[J];
         } // for (J = 0; J < j; ++J)
         F0 = sigmoid(thetai);

         error_reached = 0.5 * pow((answers[training_iterator % 4] - F0), 2);

         cout << "Iteration Number: " << training_iterator << ", Test Case: " << a[0] << " & " << a[1] <<
            "Expected Output: " << answers[training_iterator % 4] << ", Output: " << F0 << ", Error: " << error_reached << endl << endl;

         lowerOmega = answers[training_iterator % 4] - F0;
         lowerPsi = lowerOmega * sigmoidPrime(thetai);
         for (J = 0; J < j; ++J)
         {
            EPrimeJ0 = -1 * h[J] * lowerPsi;
            deltaWj0 = -1 * lambda * EPrimeJ0;
            weights0j[J] += deltaWj0;
         }

         for (J = 0; J < j; ++J)
         {
            capitalOmega = lowerPsi * weights0j[J];
            capitalPsi = capitalOmega * sigmoidPrime(thetaj);

            for (K = 0; K < k; ++K)
            {
               EPrimeJK = -1 * a[K] * capitalPsi;
               deltaWJK = -1 * lambda * EPrimeJK;
               weightsjk[J][K] += deltaWJK;
            }
         }

      } // for (training_iterator = 0; training_iterator ...

      return;
   } // void Train()

   /**
    * Runs the network using predetermined test data. Each node is calculated using the sigmoid function applied
    *    onto a dot product of the weights and the activations
    */
   void Run(double *inputValues) // TODO: Figure out how to find the error and time taken
   {
      checkNetwork();
      a = inputValues;

      for (J = 0; J < j; ++J)
      {
         thetaj = 0;
         for (K = 0; K < k; ++K)
         {
            thetaj += a[K] * weightsjk[J][K];
         } // for (K = 0; K < k; ++K)
         h[J] = sigmoid(thetaj);
      } // for (J = 0; J < j; ++J)

      thetai = 0;
      for (J = 0; J < j; ++J)
      {
         thetai += h[J] * weights0j[J];
      } // for (J = 0; J < j; ++J)
      F0 = sigmoid(thetai);

      cout << "Output: " << F0 << endl;
      return;
   } // void Run(double inputValues[])

   /**
    * Reports the results of the training or running of the network, depending on the mode the network
    *    is in training mode or not
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

         cout << "Iterations reached: " << iterations << endl;
      } // else
      return;
   } // reportResults()
}; // struct NeuralNetwork

/**
 * Main function of the program - creates and configures the network, trains it, and then runs it
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

   double** train_data = new double*[4];
   train_data[0] = new double[];
   train_data[1] = new double[];
   train_data[2] = new double[];
   train_data[3] = new double[];
   train_data[0][0] = 0;
   train_data[0][1] = 0;
   train_data[1][0]


   double train_answers[] = {0, 0, 0, 1};
   network->Train(train_data, train_answers); // Training the Network using predetermined training data
   network->reportResults();

   network->training = false; // Running the Network using test data
   double testdata[] = {0.0, 1.0}; // TODO Fix Implementation of Test and Train Data
   network->Run(testdata);
   // network->reportResults();

   return 0;
} // int main(int argc, char *argv[])
