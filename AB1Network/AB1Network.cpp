/*
 * Author: Varun Thvar
 * Date of Creation: 26 January 2023
 * Description: This is an implementation of a simple A-B-1 Network designed for simple boolean algebra problems.
 */

#include <cmath>
#include <iostream>
#include <time.h>

#define largeNumber              (int) INT_MAX
#define numberActivations        (int) 2
#define numberHiddenLayers       (int) 1
#define numberHiddenNodes        (int) 32
#define MAX_ITERATIONS           (int) 100000
#define LAMBDA                   (double) 0.3
#define RANDOM_MIN               (double) -1.5
#define RANDOM_MAX               (double) 1.5
#define ERROR_THRESHOLD          (double) (2.0 * pow(10, -4))
#define MILLISECONDS_IN_SECOND   (double) 1000.0
#define SECONDS_IN_MINUTE        (double) 60.0
#define MINUTES_IN_HOUR          (double) 60.0
#define HOURS_IN_DAY             (double) 24.0
#define DAYS_IN_WEEK             (double) 7.0

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
 * Accept a value representing seconds elapsed and print out a decimal value in easier to digest units
 * The magic numbers 1000., 60., 60., 24. and 7. need to be replaced with constants.
 *
 * Written by Dr. Eric Nelson. Modified by Varun Thvar
 */
void printTime(double seconds)
{
   double minutes, hours, days, weeks;

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
   cout << endl;
   return;
} // void printTime(double seconds)

/**
 * This is the structure of the Neural Network, which contains the following variables:
 *    numInputActivations: Number of Activations
 *    numHiddenActivations: Number of Hidden Nodes in each Hidden Layer
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
 *    train_data: Training Data (Inputs)
 *    train_answers: Training Answers (Expected Outputs)
 *    test_data: Test Data (Inputs)
 *    numCases: Number of Test Cases
 *    training: Whether or not the network is in training mode (the alternative being running mode)
 *    I, J, K: Variables used for looping through the outputs, hidden layer, and activations respectively
 *    epoch: Used to iterate over all the test cases
 *    D: Used to iterate over all the data points in a train case
 *    thetaj: Values used for calculating the Hidden Nodes - dot product of activations and corresponding weights
 *    thetai: Values used for calculating the Output - dot product of hidden layers and corresponding weights
 *    dummyError: Dummy variable used for calculating the error
 *    lowerOmega: Value of T0 - F0
 *    lowerPsi: Value of lowerOmega * sigmoidPrime(thetai)
 *    EPrimeJ0: Value of -1 * h[J] * lowerPsi -> indicates derivative of the error with respect to the weights0j
 *    deltaWj0: Value of -1 * lambda * EPrimeJ0 -> indicates the change in weights0j
 *    capitalOmega: Values of lowerPsi * weights0j[J]
 *    capitalPsi: Values of capitalOmega * sigmoidPrime(thetaj)
 *    EPrimeJK: Value of -1 * a[K] * capitalPsi -> indicates derivative of the error with respect to the weightsjk
 *    deltaWJK: Value of -1 * lambda * EPrimeJK -> indicates the change in weightsjk
 *    trainingTime: Time taken for training the network
 *    runningTime: Time taken for running the network
 *    randomize: Whether or not the network is in randomize mode
 *    keepAliveFrequency: Frequency of keep alive mode
 *    dummyStart: Dummy variable used for timing
 *    dummyEnd: Dummy variable used for timing
 *    iterations: Number of iterations taken during training
 *    error_reached: Error value reached at the end of training or running
 *    reasonForEndOfTraining: Reason for the end of training
 *
 * And the following methods:
 *   randomValue() - Outputs a random value based on the range as given in the configuration parameters
 *   setConfigurationParameters() - Sets the configuration parameters for the network based on the following parameters
 *   echoConfigurationParameters() - Outputs the configuration parameters for the network
 *   allocateArrayMemory() - Allocates memory for the activations, hidden nodes, and weights of the network
 *   populateArrays() - Populates the arrays with random values, unless the network is a 2-2-1 network, in which
 *          it manually overrides
 *   checkNetwork() - Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
 *          and the Random Number Range
 *   Train() - Trains the network using predetermined training data. The training is done using the gradient
 *          descent algorithm, which is used to minimize the error by adjusting the weights derivative of the error
 *          with respect to the weights
 *   Run() - Runs the network using predetermined test data. Each node is calculated using the sigmoid function
 *          applied onto a dot product of the weights and the activations
 *   reportResults() - Reports the results of the training or running of the network, depending on the mode
 *          the network is in training mode or not
 *
 */
struct NeuralNetwork
{
   int numOutputActivations;
   int numInputActivations;
   int numHiddenActivations;
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
   double** train_data;
   double* train_answers;
   double** test_data;
   int numCases;

   bool training;
   int I, J, K;
   int D;
   int epoch;
   double* thetaj;
   double* thetai;

   float dummyError;
   double lowerOmega;
   double lowerPsi;
   double EPrimeJ0;
   double deltaWj0;
   double* capitalOmega;
   double* capitalPsi;
   double EPrimeJK;
   double deltaWJK;

   double trainingTime;
   double runningTime;
   int keepAliveFrequency;
   bool randomize;
   time_t dummyStart;
   time_t dummyEnd;
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
                                   float errorThres, int maxIter, double min, double max,
                                   bool train, bool randomize, int keepAliveFrequency)
   {
      this->numOutputActivations = 1;
      this->numInputActivations = numAct;
      this->numHiddenLayers = numHidLayer;
      this->numHiddenActivations = numHidInEachLayer;

      this->lambda = lamb;
      this->errorThreshold = errorThres;
      this->maxIterations = maxIter;
      this->randMin = min;
      this->randMax = max;

      this->training = train;
      this->randomize = randomize;
      this->numCases = 4;
      this->keepAliveFrequency = keepAliveFrequency;
      return;
   } // void setConfigurationParameters(int numAct, int numHidLayer ...

  /**
   * Outputs the configuration parameters for the network
   */
   void echoConfigurationParameters()
   {
      cout << endl << "Echoing Configuration Parameters:" << endl;
      cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" << 1 << endl;
      cout << "Lambda Value: " << this->lambda << endl;
      cout << "Error Threshold: " << this->errorThreshold << endl;
      cout << "Maximum Number of Iterations: " << this->maxIterations << endl;
      cout << "Random Value Minimum: " << this->randMin << endl;
      cout << "Random Value Maximum: " << this->randMax << endl;
      cout << "Randomizing Weights: " << (randomize ? "True" : "False") << endl;
      return;
   } //void echoConfigurationParameters()

   /**
    * Allocates memory for the activations, hidden nodes, and weights of the network
    */
   void allocateArrayMemory()
   {
      a = new double[numInputActivations];
      h = new double[numHiddenActivations];

      weights0j = new double[numHiddenActivations];
      weightsjk = new double*[numHiddenActivations];
      for (J = 0; J < numHiddenActivations; ++J) weightsjk[J] = new double[numInputActivations];

      capitalOmega = new double[numHiddenActivations];
      capitalPsi = new double[numHiddenActivations];

      thetai = new double[numOutputActivations];
      thetaj = new double[numHiddenActivations];

      train_data = new double*[4];
      for (int i = 0; i < 4; ++i) train_data[i] = new double[2];
      train_answers = new double[4];
      test_data = new double*[4];
      for (int i = 0; i < 4; ++i) test_data[i] = new double[2];

      cout << "Allocated Memory!" << endl;
      return;
   } //void allocateArrayMemory()

   /**
    * Populates the arrays with random values, unless the network is a 2-2-1 network, in which it manually overrides
    *    the values to match a set of pre-determined values
    */
   void populateArrays(bool random)
   {
      if (randomize) // Randomizing Weights
      {
         for (J = 0; J < numHiddenActivations; ++J) weights0j[J] = randomValue();
         for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K)
            weightsjk[J][K] = randomValue();
      }
      else // Manually Overriding Values
      {
         weights0j[0] = 0.103;
         weights0j[1] = 0.23;
         for (J = 0; J < numHiddenActivations; ++J) for (int K = 0; K < numInputActivations; ++K) weightsjk[J][K] = 1;
      }

      for (J = 0; J < numHiddenActivations; ++J) thetaj[J] = 0; // Initializing thetas
      for (I = 0; I < numOutputActivations; ++I) thetai[I] = 0;

      for (J = 0; J < numHiddenActivations; ++J) capitalOmega[J] = 0; // Initializing capital Omega and Psi
      for (J = 0; J < numHiddenActivations; ++J) capitalPsi[J] = 0;

      train_data[0][0] = 0; // Initializing Training Data
      train_data[0][1] = 0;
      train_data[1][0] = 0;
      train_data[1][1] = 1;
      train_data[2][0] = 1;
      train_data[2][1] = 0;
      train_data[3][0] = 1;
      train_data[3][1] = 1;

      train_answers[0] = 0; // Initializing Training Answers
      train_answers[1] = 1;
      train_answers[2] = 1;
      train_answers[3] = 1;

      test_data[0][0] = 0; // Initializing Test Data
      test_data[0][1] = 0;
      test_data[1][0] = 0;
      test_data[1][1] = 1;
      test_data[2][0] = 1;
      test_data[2][1] = 0;
      test_data[3][0] = 1;
      test_data[3][1] = 1;

      cout << "Populated Arrays!" << endl;
      return;
   } //void populateArrays()

   /**
    * Outputs the Network Type, Lambda Value, Error Threshold, Maximum Number of Iterations,
    *    and the Random Number Range. To be used before training and/or running.
    */
   void checkNetwork()
   {
      if (!training)
         cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" << 1 << endl;
      if (training)
      {
         cout << "Training the Network!" << endl;
         cout << "Network Type: " << numInputActivations << "-" << numHiddenActivations << "-" << 1 << endl;
         cout << "Lambda Value: " << this->lambda << endl;
         cout << "Error Threshold: " << this->errorThreshold << endl;
         cout << "Maximum Number of Iterations: " << this->maxIterations << endl;
         cout << "Random Value Minimum: " << this->randMin << endl;
         cout << "Random Value Maximum: " << this->randMax << endl << endl;
      }
      return;
   }

   /**
    * Trains the network using predetermined training data using the gradient descent algorithm, which is used to
    *    minimize the error by adjusting the weights derivative of the error with respect to the weights
    */
   void Train()
   {
      time(&dummyStart);

      checkNetwork();

      error_reached = largeNumber;
      dummyError = 0;
      epoch = 0;

      while (epoch < maxIterations && error_reached > errorThreshold)
      {
         error_reached = 0;
         for (D = 0; D < 4; ++D)
         {
            a = train_data[D];

            thetaj[J] = 0;
            for (J = 0; J < numHiddenActivations; ++J)
            {
               thetaj[J] = 0;
               for (K = 0; K < numInputActivations; ++K)
               {
                  thetaj[J] += a[K] * weightsjk[J][K];
               } // for (K = 0; K < numInputActivations; ++K)
               h[J] = sigmoid(thetaj[J]);
            } // for (J = 0; J < numHiddenActivations; ++J)

            thetai[0] = 0;
            for (J = 0; J < numHiddenActivations; ++J)
            {
               thetai[0] += h[J] * weights0j[J];
            } // for (J = 0; J < numHiddenActivations; ++J)
            F0 = sigmoid(thetai[0]);

            dummyError = 0.5 * pow((train_answers[D] - F0), 2);
            error_reached += dummyError;

            lowerOmega = train_answers[D] - F0;
            lowerPsi = lowerOmega * sigmoidPrime(thetai[0]);
            for (J = 0; J < numHiddenActivations; ++J)
            {
               EPrimeJ0 = -1 * h[J] * lowerPsi;
               deltaWj0 = -1 * lambda * EPrimeJ0;
               weights0j[J] += deltaWj0;
            } // for (J = 0; J < numHiddenActivations; ++J)

            for (J = 0; J < numHiddenActivations; ++J)
            {
               capitalOmega[J] = lowerPsi * weights0j[J];
               capitalPsi[J] = capitalOmega[J] * sigmoidPrime(thetaj[J]);

               for (K = 0; K < numInputActivations; ++K)
               {
                  EPrimeJK = -1 * a[K] * capitalPsi[J];
                  deltaWJK = -1 * lambda * EPrimeJK;
                  weightsjk[J][K] += deltaWJK;
               } // for (K = 0; K < numInputActivations; ++K)
            } // for (J = 0; J < numHiddenActivations; ++J)
         } // for (D = 0; D ...

         error_reached /= numCases;

         if (epoch % keepAliveFrequency == 0) // TODO Destroy this
         {
            cout << "Iteration Number: " << epoch << ", Test Case: " << a[0] << " & " << a[1] <<
               ", Expected Output: " << train_answers[D] << ", Output: " << F0 << ", Error: " << dummyError << endl << endl;
         } // if (epoch % keepAliveFrequency == 0 && keepAlive)

         ++epoch;
      } // while (epoch < maxIterations && error_reached > errorThreshold)

      if (epoch == maxIterations) reasonForEndOfTraining = "Maximum Number of Iterations Reached";
      else reasonForEndOfTraining = "Error Threshold Reached";

      trainingTime = double(time(&dummyEnd) - dummyStart);
      iterations = epoch;

      return;
   } // void Train()

   /**
    * Runs the network using predetermined test data. Each node is calculated using the sigmoid function applied
    *    onto a dot product of the weights and the activations
    */
   double Run(double *inputValues)
   {
      training = false;
      checkNetwork();
      a = inputValues;

      time(&dummyStart);

      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetaj[J] = 0;
         for (K = 0; K < numInputActivations; ++K)
         {
            thetaj[J] += a[K] * weightsjk[J][K];
         } // for (K = 0; K < numInputActivations; ++K)
         h[J] = sigmoid(thetaj[J]);
      } // for (J = 0; J < numHiddenActivations; ++J)

      thetai[0] = 0;
      for (J = 0; J < numHiddenActivations; ++J)
      {
         thetai[0] += h[J] * weights0j[J];
      } // for (J = 0; J < numHiddenActivations; ++J)
      F0 = sigmoid(thetai[0]);

      time(&dummyEnd);
      runningTime = double(dummyEnd - dummyStart);

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
         printTime(trainingTime);
         cout << "Error Reached: " << error_reached << endl;
         cout << "Iterations reached: " << iterations << endl << endl;
      } // if (training)
      return;
   } // reportResults()
}; // struct NeuralNetwork


/**
 * Accepts input from the user and runs the network using the input values. Specific to Boolean Algebra
 */
void testingData(NeuralNetwork* network)
{
   for (int i = 0; i < 4; ++i)
   {
      cout << "Running the Network with Test Data: " << network->test_data[i][0] << " & " << network->test_data[i][1] << endl;
      cout << network->Run(network->test_data[i]) << endl << endl;
   } // for (int numOutputActivations = 0; numOutputActivations < 4; ++numOutputActivations)
   return;
} // void testingData(NeuralNetwork* network)

/**
 * Main function of the program - creates and configures the network, trains it, and then runs it for all test cases
 */
int main(int argc, char *argv[])
{
   NeuralNetwork* network = new NeuralNetwork(); // Creating and Configurating the Network based on pre-determined constants and designs
   network->setConfigurationParameters(numberActivations, numberHiddenLayers,
      numberHiddenNodes, LAMBDA, ERROR_THRESHOLD, MAX_ITERATIONS,
      RANDOM_MIN, RANDOM_MAX, true, true, 1000);
   network->echoConfigurationParameters();
   cout << endl;

   network->allocateArrayMemory(); // Allocating Arrays in Network
   network->populateArrays(true);
   cout << endl;

   network->Train(); // Training the Network using predetermined training data
   network->reportResults();

   network->training = false; // Running the Network using test data
   testingData(network);
   network->reportResults();

   delete network; // Deletes all objects and pointers used in the program

   return 0;
} // int main(int argc, char *argv[])
