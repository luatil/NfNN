@echo off

set opts=-FC -GR- -EHa- -nologo -Zi
set code=tests
@set INCLUDES=/Ilib
@set OUT_DIR=build

if not exist %OUT_DIR% mkdir %OUT_DIR%
pushd %OUT_DIR%
 
REM build tests
cl %INCLUDES% %opts% ..\tests\tests.c -Fetests.exe -I%includes%

REM build examples
cl %INCLUDES% %opts% ..\examples\xor\mlp\src\xor.c -Fexor.exe -I%includes%

REM build examples - mnist dataloader
cl %INCLUDES% %opts% ..\examples\mnist\mnist_single_device\src\mnist_test_dataloader.c -Femnist_test_dataloader.exe -I%includes%

REM build examples - mnist single device
cl %INCLUDES% %opts% ..\examples\mnist\mnist_single_device\src\mnist_single_device.c -Femnist_single_device.exe -I%includes%

REM build examples - mnist sync parameter server
cl %INCLUDES% %opts% ..\examples\mnist\mnist_sync_parameter_server\src\mnist_sync_parameter_server.c -Femnist_sync_parameter_server.exe -I%includes%

REM build examples - mnist async parameter server
cl %INCLUDES% %opts% ..\examples\mnist\mnist_async_parameter_server\src\mnist_async_parameter_server.c -Femnist_async_parameter_server.exe -I%includes%
popd


