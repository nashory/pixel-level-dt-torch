opt = {
   dataset = 'folder',      
   batchSize = 128,
   loadSize = 64,
   ngf = 96,               -- #  of gen filters in first conv layer
   ndf = 96,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   nTotalEpoch = 10000,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 0,                -- gpu = -1 is CPU mode. gpu=X is GPU mode on GPU X (starts from index 0)
   name = 'exp1',
   noise = 'normal',       -- uniform / normal
   optimizer = 'sgd', 
   load_cp = 0,
   display_every = 5,      -- dispay every X iterations
   logging_every = 1,      -- log every X iteartions
   save_png_every = 10,    -- save png files every X iterations

   display_ip = '10.108.23.11',
   display_port = 8889,
}

return opt
