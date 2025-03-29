def train_gan(D, G, D_solver, G_solver, noise=args.Ydim, num_epochs=10):
    iter_count = 0 
    l2_Acc = val_G(noise)
    Best_acc = l2_Acc.copy()
    
    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(loader_label):
            if x.size(0) != args.batch:
                continue
    
            eta = Variable(sample_noise(x.size(0), noise))
            
            d_input = torch.cat([x, y.view(args.batch, 1)], dim=1)
            g_input = torch.cat([x, eta], dim=1)
            
            #train D
            D_solver.zero_grad()
            logits_real = D(d_input)
            
            fake_y = G(g_input).detach()
            fake_images = torch.cat([x, fake_y.view(args.batch, 1)], dim=1)
            logits_fake = D(fake_images)
            
            penalty = calculate_gradient_penalty(D, logits_real, fake_images, device)
            d_error = discriminator_loss(logits_real, logits_fake) + 10 * penalty
            d_error.backward() 
            D_solver.step()
             
            # train G
            G_solver.zero_grad()
            fake_y = G(g_input)
            fake_images = torch.cat([x, fake_y.view(args.batch, 1)], dim=1)
            logits_fake = D(fake_images)
            
            g_output = torch.zeros([50, args.batch])
            for i in range(50):
                eta = sample_noise(x.size(0), noise)
                g_input = torch.cat([x, eta], dim=1)
                g_output[i] = G(g_input).view(x.size(0))
            
            g_error = 0.9 * generator_loss(logits_fake) + 0.1 * l2_loss(g_output.mean(dim=0), y)
            g_error.backward()
            G_solver.step()
            
            if(iter_count > 2500):    
                if(iter_count % 100 == 0):
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_error.item(), g_error.item()))
                    l2_Acc = val_G(noise)
                    if l2_Acc < Best_acc:
                        print('################## save G model #################')
                        Best_acc = l2_Acc.copy()
                        torch.save(G, './M2d5/WGR-G-m' + str(noise) + '.pth')
                        #torch.save(D, './M2d5/WGR-D-m' + str(noise) + '.pth')
            iter_count += 1
