# /hashing_testing/ project, which uses AVX512 to implement different approaches of LinearProbing algorithm for hashbased group count 

! Important for executing on Intel DevCloud:    
Do not run the project on the login node login-2:~$! This can lead to the execution simply being stopped 
after an indefinite period of time (probably due to a lack of resources in terms of memory or reserved computing time). 

The following commands should therefore be used (interactive execution):

(1)	`source /data/intel_fpga/devcloudLoginToolSetup.sh`
(2) `devcloud_login`
(3) select Option "6) Enter Specific Node Number"
(4) select free node of type: "Nodes with no attached hardware:" - e.g. s001-n070

(5) switch back to the working directory

(6) Compile Code
    `./build.sh`

(7) Execute
    `./execute.sh`

----------------

@TODO:  steps for execution with a batchjob

