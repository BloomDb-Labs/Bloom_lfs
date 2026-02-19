llama_crate/
doc/
|  ├─ documentation
|  ├─ reference paper
src/
│  ├─ lib.rs
│  ├─ benching/
│  ├─ llama          # Full llama subsystem
│  ├─ mapping_table  # Generic Mapping Table using Dashmap, Aliases             
│  ├─ data_objects   # Pages, Deltas, Annotations, Physical Addresses              
│  ├─ caching_layer/   
|  |    ├─ policy
|  |    ├─ cache               
│  |─ config     
│  ├─ transactions
|  ├─ recovery                 
|  ├─ storage_layer/
|  |    ├─ log_storage
|  |    ├─ flush_buffers
|  |    ├─ input_output # io_uring
└─ 