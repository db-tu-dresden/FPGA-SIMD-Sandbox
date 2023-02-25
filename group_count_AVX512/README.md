# Own AVX512-Implementation of group_count algorithm with LinearProbing approach

(1) Compile Code

    (a) `cmake .` or `cmake -DCMAKE_CXX_COMPILER=... .`

    (b) `make`

(2) Execute
`bin/main`


-   Within Intel Devcloud: DON'T execute project on login-2 node!
-   Login interactive on a computing node:
        (1) `source /data/intel_fpga/devcloudLoginToolSetup.sh`
        (2) `devcloud_login`
        (3) select option 6
        (4) select a node of type "Nodes with no attached hardware"