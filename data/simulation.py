# =============================================================================
# Univerate Y
# =============================================================================
# generate data from SM1
# training data
train_X = torch.randn([args.train,args.Xdim])
train_eps = torch.randn([args.train]) 
train_Y = train_X[:,0]**2 + torch.exp(train_X[:,1]+train_X[:,2]/3) + torch.sin(train_X[:,3]+train_X[:,4]) + train_eps

# validation data 
val_X = torch.randn([args.val,args.Xdim]) 
val_eps = torch.randn([args.val]) 
val_Y = val_X[:,0]**2 + torch.exp(val_X[:,1]+val_X[:,2]/3) + torch.sin(val_X[:,3]+val_X[:,4]) + val_eps

# test dat
test_X = torch.randn([args.test,args.Xdim]) 
test_eps = torch.randn([args.test]) 
test_Y = test_X[:,0]**2 + torch.exp(test_X[:,1]+test_X[:,2]/3) + torch.sin(test_X[:,3]+test_X[:,4]) + test_eps

# generate data from M1
# training data
train_X = torch.randn([args.train,args.Xdim])
train_eps = torch.randn([args.train]) 
train_Y = train_X[:,0]**2 + torch.exp(train_X[:,1]+train_X[:,2]/3) + train_X[:,3] - train_X[:,4] + (0.5 + train_X[:,1]**2/2 + train_X[:,4]**2/2)*train_eps

# validation data 
val_X = torch.randn([args.val,args.Xdim]) 
val_eps = torch.randn([args.val]) 
val_Y = val_X[:,0]**2 + torch.exp(val_X[:,1]+val_X[:,2]/3) + val_X[:,3] - val_X[:,4] + (0.5 + val_X[:,1]**2/2 + val_X[:,4]**2/2)*val_eps

# test data
test_X = torch.randn([args.test,args.Xdim]) 
test_eps = torch.randn([args.test]) 
test_Y = test_X[:,0]**2 + torch.exp(test_X[:,1]+test_X[:,2]/3) + test_X[:,3] - test_X[:,4] + (0.5 + test_X[:,1]**2/2 + test_X[:,4]**2/2)*test_eps
    
# generate data from M2




# Multivariate Y
