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

The example mnist_single_device is an example of optimization
using only a single device.


