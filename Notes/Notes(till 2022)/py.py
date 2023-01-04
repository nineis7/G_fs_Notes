def FFT(P):
    n=len(P)
    if n==1:
        return P
    
    w=math.exp(2j*pi/n)
    
    Pe,Po = [P[0]:2:P[n-2], P[1]:2:P[n-1]]
    ye,yo = FFT(Pe), FFT(Po)
    y = [0] * n
    for j in range(n/2):
        y[j] = ye[j] + w^j*yo[j]
        y[j+n/2] = ye[j]-w^j*yo[j]
    return y



from numpy import mgrid, sum

def moments2e(image):
  assert len(image.shape) == 2 # only for grayscale images        
  y,x = mgrid[:image.shape[0],:image.shape[1]]
  moments = {}
  moments['mean_x'] = sum(x*image)/sum(image)
  moments['mean_y'] = sum(y*image)/sum(image)
          
  # raw or spatial moments
  moments['m00'] = sum(image)
  moments['m01'] = sum(y*image)
  moments['m10'] = sum(x*image)
  moments['m11'] = sum(y*x*image)
  moments['m02'] = sum(y**2*image)
  moments['m20'] = sum(x**2*image)
  moments['m12'] = sum(x*y**2*image)
  moments['m21'] = sum(x**2*y*image)
  moments['m03'] = sum(x**3*image)
  moments['m30'] = sum(y**3*image)
  
  # central moments
  moments['mu11'] = sum((x-moments['mean_x'])*(y-moments['mean_y'])*image)
  moments['mu02'] = sum((y-moments['mean_y'])**2*image) # variance
  moments['mu20'] = sum((x-moments['mean_x'])**2*image) # variance
  moments['mu12'] = sum((x-moments['mean_x'])*(y-moments['mean_y'])**2*image)
  moments['mu21'] = sum((x-moments['mean_x'])**2*(y-moments['mean_y'])*image) 
  moments['mu03'] = sum((y-moments['mean_y'])**3*image) 
  moments['mu30'] = sum((x-moments['mean_x'])**3*image) 


        
  # central standardized or normalized or scale invariant moments
  moments['nu11'] = moments['mu11'] / sum(image)**(2/2+1)
  moments['nu12'] = moments['mu12'] / sum(image)**(3/2+1)
  moments['nu21'] = moments['mu21'] / sum(image)**(3/2+1)
  moments['nu20'] = moments['mu20'] / sum(image)**(2/2+1)
  moments['nu03'] = moments['mu03'] / sum(image)**(3/2+1) # skewness
  moments['nu30'] = moments['mu30'] / sum(image)**(3/2+1) # skewness
  return moments
