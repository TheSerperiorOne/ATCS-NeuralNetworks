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
#define RANDOM_MIN ((double) -1.5)
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
 * This is the structure of the Neural Network, which contains the following variables:
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
 *    epoch: Used to iterate over all the test cases
 *    data_iterator: Used to iterate over all the data points in a train case
 *    thetaj: Value used for calculating the Hidden Nodes
 *    thetai: Value used for calculating the Output
 *    dummyError: Dummy variable used for calculating the error
 *    lowerOmega: Value of T0 - F0
 *    lowerPsi: Value of lowerOmega * sigmoidPrime(thetai)
 *    EPrimeJ0: Value of -1 * h[J] * lowerPsi -> indicates derivative of the error with respect to the weights0j
 *    deltaWj0: Value of -1 * lambda * EPrimeJ0 -> indicates the change in weights0j
 *    capitalOmega: Value of lowerPsi * weights0j[J]
 *    capitalPsi: Value of capitalOmega * sigmoidPrime(thetaj)
 *    EPrimeJK: Value of -1 * a[K] * capitalPsi -> indicates derivative of the error with respect to the weightsjk
 *    deltaWJK: Value of -1 * lambda * EPrimeJK -> indicates the change in weightsjk
 *    training_time: Time taken for training the network
 *    running_time: Time taken for running the network
 *    dummyStart: Dummy variable used for timing
 *    dummyEnd: Dummy variable used for timing
 *    iterations: Number of iterations taken during training
 *    error_reached: Error value reached at the end of training or running
 *    reasonForEndOfTraining: Reason for the end of training
 *
 * And the following methods: // TODO: add methods to documentation
 *
 */
struct NeuralNetwork
{
   int k;
   int j;
   int numHiddenLayers;
   double lambda;
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
   int epoch;
   int data_iterator;
   double thetaj;
   double thetai;

   float dummyError;
   double lowerOmega;
   double lowerPsi;
   double EPrimeJ0;
   double deltaWj0;
   double capitalOmega;
   double capitalPsi;
   double EPrimeJK;
   double deltaWJK;

   int training_time;
   int running_time;
   long dummyStart;
   long dummyEnd;
   int iterations;
   float error_reached;
   string reasonForEndOfTraining;

   /**
    * Outputs a random value based on the range as given in the configuration parameters
    */
   double randomValue()
   {
      return ((double) rand() / (RAND_MAX)) * (randMax - randMin) + randMin;
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
   void setConfigurationParameters(int numAct, int numHidLayer, int numHidInEachLayer, double lamb,
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
   void populateArrays(bool random)
   {
      for (int J = 0; J < j; ++J) weights0j[J] = randomValue();
      for (int J = 0; J < j; ++J) for (int K = 0; K < k; ++K) weightsjk[J][K] = randomValue();
      cout << "Populated Arrays!" << endl;
      return;
   } //void populateArrays()

   /**
    * Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
    *    and the Random Number Range. To be used before training and/or running.
    */
   void checkNetwork()
   {
      if (training) cout << "Training the Network!" << endl;
      else cout << "Running the Network!" << endl;
      cout << "Network Type: " << k << "-" << j << "-" << 1 << endl;
      cout << "Lambda Value: " << this->lambda << endl;
      cout << "Error Threshold: " << this->errorThreshold << endl;
      cout << "Maximum Number of Iterations: " << this->maxIterations << endl;
      cout << "Random Value Minimum: " << this->randMin << endl;
      cout << "Random Value Maximum: " << this->randMax << endl << endl;

      return;
   }

   /**
    * Trains the network using predetermined training data
    */
   void Train(double** data, double* answers) // Fix Time Implementation
   {
      training_time = 0;
      dummyStart = clock();
      checkNetwork();
      error_reached = pow(2, 20);
      dummyError = 0;

      for (epoch = 0; epoch < maxIterations && error_reached > errorThreshold; ++epoch)
      {
         error_reached = 0;
         cout << "Gotten Here!";
         for (data_iterator = 0; data_iterator < 4; ++data_iterator)
         {
            a = data[data_iterator];

            thetaj = 0;
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

            dummyError = 0.5 * pow((answers[data_iterator] - F0), 2);
            error_reached += dummyError;

            cout << "Iteration Number: " << epoch << ", Test Case: " << a[0] << " & " << a[1] <<
               ", Expected Output: " << answers[data_iterator] << ", Output: " << F0 << ", Error: " << dummyError << endl << endl;

            lowerOmega = answers[data_iterator] - F0;
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
            } // for (J = 0; J < j; ++J)
         } // for (data_iterator = 0; data_iterator ...
         error_reached /= sizeof(data)/sizeof(data[0]);
      } // for (epoch = 0; epoch ...

      if (epoch == maxIterations) reasonForEndOfTraining = "Maximum Number of Iterations Reached";
      else reasonForEndOfTraining = "Error Threshold Reached";

      training_time = clock() - dummyStart;
      iterations = epoch;

      return;
   } // void Train()

   /**
    * Runs the network using predetermined test data. Each node is calculated using the sigmoid function applied
    *    onto a dot product of the weights and the activations
    */
   double Run(double *inputValues)
   {
      checkNetwork();
      a = inputValues;

      dummyStart = clock();

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

      running_time = clock() - dummyStart;

      return F0;
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
 * Accepts input from the user and runs the network using the input values. Specific to Boolean Algebra
 */
void userInputRunSection(NeuralNetwork* network)
{
   double* testdata = new double[2];
   cout << "Input 1st Value for Network: " << endl;
   cin >> testdata[0];
   cout << "Input 2nd Value for Network: " << endl;
   cin >> testdata[1];
   cout << "Output: " << endl << endl <<  network->Run(testdata) << endl << endl;
} // void userInputRunSection(NeuralNetwork* network)

/**
 * Main function of the program - creates and configures the network, trains it, and then runs it
 */
int main(int argc, char *argv[])
{
   /**
    * Creating and Configurating the Network based on pre-determined constants and designs
    */
   NeuralNetwork* network = new NeuralNetwork();
   network->setConfigurationParameters(2, 1, 20, LAMBDA,
      ERROR_THRESHOLD, MAX_ITERATIONS, RANDOM_MIN, RANDOM_MAX, true);
   network->echoConfigurationParameters();
   cout << endl;

   network->allocateArrayMemory(); // Allocating Arrays in Network
   network->populateArrays(true);

   cout << endl;

   double** train_data = new double*[4]; // Setting up training data
   train_data[0] = new double[2];
   train_data[1] = new double[2];
   train_data[2] = new double[2];
   train_data[3] = new double[2];
   train_data[0][0] = 0;
   train_data[0][1] = 0;
   train_data[1][0] = 0;
   train_data[1][1] = 1;
   train_data[2][0] = 1;
   train_data[2][1] = 0;
   train_data[3][0] = 1;
   train_data[3][1] = 1;

   double train_answers[] = {0, 1, 1, 1}; // Setting up training answers

   network->Train(train_data, train_answers); // Training the Network using predetermined training data
   network->reportResults();

   network->training = false; // Running the Network using test data
   userInputRunSection(network);
   network->reportResults();

   delete network;
   delete[] train_data;

   return 0;
} // int main(int argc, char *argv[])
