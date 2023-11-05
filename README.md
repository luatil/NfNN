# NfNN

NfNN (Networking for Neural Networks) is an automatic differentiation
library for distributed environments written in C. It also implements
some ergonomics for working with machine learning specific problems.

## Compiling tests and examples

To compile all tests and examples run the following command (.sh for
linux and .bat for windows)

```bash
./build_all.sh
```

## Tests

The build_all script builds both examples and tests. To run the tests
to assure that the library is properly working you can run the following
commands:

```bash
cd build
./tests
```

## Examples

Some examples of library use are also included in the repository.
The commands assume you are in the NfNN/build directory.

## XOR

The XOR example is one of the most simple examples of using a MLP.

To run the XOR example you should run:

```bash
./xor
```

## MNIST

### Downloading the datasets

To run the MNIST examples (single and distributed) you first need
to download the MNIST dataset. To do that you can run:

```bash
cd ..
./download_dataset.sh
```

This requires that you have both wget and gzip installed in your system.

### Testing the dataset download

To test that the dataset downloaded properly you can run

```bash
cd build
./mnist_test_dataloader 73
```

Here the 73 can be replaced by a SEED. If the dataset was downloaded
correctly and the binary is correctly compiled you should see MNIST
examples printed on the terminal.

```
Label: 5
                            
                            
                            
                       .:-  
                   :+#%%%*  
                :=#@%#+:    
             .=%%%%+.       
            =%%*-:          
          :#%+              
         -%%-               
         #%.                
         +%*                
          =%#.              
           :%%.             
            :%#             
             -%*            
              *%.           
      .      -%#            
      %%%*=*#@%:            
      .-*%%%%*:             
           .                
```

### Single device SGD

The mnist_single_device is an example of optimization
using only a single device.

To run the example you can run:

```bash
./mnist_single_device
```

This should optimize a MLP over 5 epochs and achieve
around 78% validation accuracy.

The source code for this example can be found in 
NfNN/examples/mnist/mnist_single_device/src/mnist_single_device.c

The example was developed with the intention of showing the raw 
library being used, therefore ergonomics such as Forward function
are not implemented.


### Syncronous centralized optimization

Consider different terminals A and B. Then for testing on
localhost you can run.

```bash
# On A
./mnist_sync_parameter_server --server
```

Starts a parameter server on localhost over a predefined port. To start
a worker you will need to run:

```bash
# On A
./mnist_sync_parameter_server --worker
```

This should start the exchange of parameters over both processes.
The worker should run for 5000 iterations.

To get more information and change the default hyper-parameters for this
example consider running the following command:

```bash
./mnist_sync_parameter_server --help
```

To run the examples over a non local network you will also need to supply
the --ip argument. Consider machine A has a public IP of 8.8.8.8 so you
can run:

```bash
# On A
./mnist_sync_parameter_server --server --ip 0.0.0.0
```

To start the parameter server. Now to start the worker you will need to
supply the IP address of machine A.

```bash
# On B
./mnist_sync_parameter_server --worker --ip 8.8.8.8
```

This should start the exchange of parameters over both machines. Notice
that the optimization can take a long time if there is a high latency /
low throughput between machines A and B.

### Assyncronous centralized optimization

The assyncronous centralized optimization follows the same
design has the syncronous version. The only change required
is to run the binary ./mnist_async_parameter_server instead of
./mnist_sync_parameter_server

### Multiple workers

To train with multiple workers instead of a single worker you can
use the following on the server:

```bash
# On A
./mnist_sync_parameter_server --server --ip 0.0.0.0 --workers 2
```

Here 2 indicates that the server will expect to connect to 2 
distinct workers before starting the optimization.

