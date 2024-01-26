//
// Created by Varun Thvar on 1/26/24.
//

#ifndef A_B_1_NETWORK_H
#define A_B_1_NETWORK_H



class A_B_1_Network {
   private:
      int inputSize;
      double inputs[];
      int hiddenSize;
      double hiddens[];
      int outputSize;
      double outputs[];
      double weights_1_2[][];
      double weights_2_3[][];
      double randrange(double max, double min)
      {
         return rand()%(max-min + 1) + min;
      }
   public:

      void setConfigurationParameters();
      void echoConfigurationParameters();
      void allocateArraysMemory();
      void populateArrays();
      void train();
      void run();
};



#endif //A_B_1_NETWORK_H
