import numpy as n
import math as m
import scipy.fftpack as sf
import scipy.signal as ss

#from PythonPhasing import tifffile as tiff

# Trying a homebrewed Richardson Lucy deconv
####################################################################
def rldeconv(data1, data2, niter):
  #in traditional rldeconv speak
  #deconvolve data2 from data1 to make output
  #in our speak, output is what makes data2(computed) into data1 (measured)
 
  #tiff.imsave("data1.tif", n.transpose(data1.astype(n.float32)))
  #tiff.imsave("data2.tif", n.transpose(data2.astype(n.float32)))
 
  #copy data1 to output
  output=data1.copy()
  ccdata2=data2.flat[::-1]
  ccdata2.shape = data2.shape
  data2sum=n.sum(data2)
  err=1
  for i in range(niter):    
    prevoutput=output.copy()
    # convolve output with data2 call it tmpout
    tmpout=ss.fftconvolve(output, data2,"same")/data2sum
  # tmpout = data1/tmpout
    tmpout = n.where(tmpout>0.0, data1/tmpout, 0.0)
  # convolve tmpout with conjg(data2)? or data2[-x]? call it tmpout2
    output *= ss.fftconvolve(tmpout, ccdata2, "same")/data2sum 
  # output *= tmpout2
    err=n.sum( abs(output-prevoutput) )/n.sum(output)
    print("deconv err", err)
  return output

####################################################################
def myconvolve(data1, data2):
  return sf.ifftn( sf.fftn(data1)*sf.fftn(data2) )

# return the center of mass of the array
# copied from Jesse Clarke
####################################################################
def center_of_mass(arr):
  tot=n.sum(arr)
  dims=arr.shape
  xyz=[]
  griddims=[]
  for d in dims:
    #griddims.append( slice(-d/2+1,d/2+1) )
    griddims.append( slice(0,d) )
  grid=n.ogrid[ griddims ] 
  for g in grid:
    xyz.append( n.sum(arr*g)/tot ) 
  return xyz

# returns the conjugate centrosymmetric array
#copied from Jesse Clarke
####################################################################
def conj_reflect(array):
  F=sf.fftn(array)
  return ifftn(conj(F))


####################################################################
def box_conv_fft(arr, sigs):
  dims=arr.shape
  if len(sigs) != len(dims):
    return None
  tot=n.sum(arr)
  planes = []
  d=0
  for s in sigs:
    plane1=[0]*len(dims)
    plane2=[0]*len(dims)
    plane1[d]=-s
    plane2[d]=s
    planes.append(plane1)
    planes.append(plane2)
    d+=1

  abox=make_poly( dims, planes)
  arrk=sf.fftn(arr)
  aboxk=sf.fftn(abox)
  convag=sf.ifftshift(sf.ifftn(arrk*aboxk))
  return convag*tot/n.sum(convag)

#based on Jesse Clarke
#save an fft by creating guassian in ft space
####################################################################
def gauss_conv_fft(arr, sigs):
  dims=arr.shape
  if len(sigs) != len(dims):
    return None
  tot=n.sum(arr)
  sigk=n.array(dims)/2.0/n.pi/sigs
  print(sigk, "sigk")
  #gaussian needs to be wrap around to match the ft of arr
  gk=sf.fftshift(gauss(dims, sigk))

  arrk=sf.fftn(arr)
  convag=sf.ifftn(arrk*gk)
  convag=n.where(convag<0.0, 0.0, convag)
  convag*=tot/n.sum(convag)

  return convag


####################################################################
def AddPolyCen(array, center, planes):
  dims=array.shape

  griddims=[]
  for d in dims:
    griddims.append( slice(0,d) )
#   griddims.append( slice(-d/2+1,d/2+1) )
  grid=n.ogrid[ griddims ] 

  for plane in planes:
    sum1=n.zeros(dims)
    sum2=0
#    sum1=plane[0]*(grid[0]-center[0]) + plane[1]*(grid[1]-center[1])
#    sum2=plane[0]**2 + plane[1]**2
    for d in range(len(dims)):
      sum1+=plane[d]*(grid[d]-center[d])
      sum2+=plane[d]**2 
    array += (sum1 <= sum2) * 1

#  for plane in planes:
#    array += ( (plane[0]*(grid[0]-center[0]) +
#                plane[1]*(grid[1]-center[1]) ) <= \
#             (plane[0]**2 + plane[1]**2)) * 1

  return ((array >= len(planes))*1).astype(array.dtype)

#dimensionally agnostic gaussian
####################################################################
def gauss(dims, sigs):
  if (len(sigs) != len(dims)):
    return None

  griddims=[]
  for d in dims:
    griddims.append( slice(-d/2+1,d/2+1) )
  grid=n.ogrid[ griddims ] 

  g=n.ones(dims) 
  d=0 
  for sig in sigs:
    if sig==0:
      g*=(grid[d]==0).astype(n.float)
    else:
      g*=n.exp(-0.5*grid[d]*grid[d]/sig**2)
    d+=1

  return g


# This should work for any shape array?
# May be a little complicated and uses a bit of extra memory, maybe.
####################################################################
def Bin(Array, Binsizes):
  odims=Array.shape
  ndims=[]
  Binsizes=list(Binsizes)
  if len(Binsizes) < len(odims):
    while len(odims) > len(Binsizes):
      Binsizes.append(1)
        
# print Binsizes
  #playing interger math tricks
  n=0
  for d in odims:
    ndims.append( d/Binsizes[n]*Binsizes[n] )
    n+=1

  def getindex(n, binsize, start, ndim):
    ind=[]
    for i in range(ndim):
      if i < n:
        ind.append(slice(0,ndims[i]))
      elif i==n:
        #playing integer math games....
        ind.append(slice(a1, odims[i]/binsize*binsize, binsize))
      else:
        ind.append(slice(0,odims[i]))
    return ind

  # Bin each dim separately using a an indexing trick
  arrs=[Array.copy()]
  n=0
  for b in Binsizes:
    for a1 in range(b):
#     print "ind", a1, b, getindex(n,b,a1,len(odims))
      if a1==0:
        arrs.append( arrs[0][getindex(n,b,a1,len(odims))] )
      else:
        arrs[1]+=arrs[0][getindex(n,b,a1,len(odims))]
    n+=1
    del arrs[0]
    
# print arrs
  return arrs[0]

# returns a new array with the passed array at the center.
# if orig array is in wrap around order you will have problems.
####################################################################
def ZeroPad(array, Padding):
    dims=list(array.shape)
    origdims=dims[:]
#    print len(dims), len(Padding), array.shape
    if len(dims) > len(Padding):
        raise(Exception, "Padding past to ZeroPad must have same or greater \
                            number of dimensions as the array")
    if len(Padding) > len(dims):
        while len(dims) < len(Padding):
            dims.append(1)
        array.shape=tuple(dims)
        
    origslice=[]
    for dim in dims:
            origslice.append( slice(0,dim, None) )
            
    newshape=[]
    newsize=1
    insertslice=[]
    for i in range(len(dims)):
        size=dims[i]+Padding[i]
        newsize = newsize*size
        newshape.append( size )
#        print size/2, dims[i]/2
        start=size/2 - dims[i]/2
        end=size/2 + int(n.ceil(dims[i]/2.))
#        print i,"sizedims",size, dims[i]
        if start==end and start>0:
            start=start-1
#        print i,"startend", start, end
        insertslice.append( slice(start,end,None) )
    newarray=n.zeros( newsize )
    newarray.shape=( tuple(newshape) )
#    print "newshape", newshape
#    print "shape:",array.shape
#    print "slice", insertslice
#    print "origslice", origslice
#    print "origdims", origdims
    
    newarray[tuple(insertslice)]=array[tuple(origslice)]
    #print "slice"
    #print newarray[tuple(insertslice)]
    #print "array"
    #print array
    array.shape=tuple(origdims)
    return newarray.copy()
    
    
def Crop(array, CropX, CropY, CropZ):
    #just use python slicing!
    pass
    
# pass the FT of the fftshifted array you want to shift
# you get back the actual array, not the FT.
####################################################################
def Shift(arr, Shifty):
    dims = arr.shape
    if len(dims) < len(Shifty):
      raise(Exception, "Shift past must have same number of dimensions as the array")

    rootN = m.sqrt(arr.size)
    #holy freaking shite my christ, scipy does normalized ffts!
    ftarr = sf.fftn(arr) #* rootN
    r=[]
    for d in dims:
      r.append(slice(int(n.ceil(-d/2.)), int(n.ceil(d/2.)), None))
#     print slice(int(n.ceil(-d/2.)), int(n.ceil(d/2.)))

    idxgrid = n.mgrid[r]

    for d in range(len(dims)):
#     print "idx", idxgrid[d].shape
      ftarr *= n.exp(-1j*2*n.pi*Shifty[d]*sf.fftshift(idxgrid[d])/float(dims[d]))

    shiftedarr = sf.ifftn(ftarr) #* rootN
    #shiftedarr = sf.ifftshift(sf.ifftn(arr)) * rootN
    return shiftedarr


####################################################################
def MakePoly(dims, planes):
  return make_poly(dims, planes)

####################################################################
def make_poly(dims, planes):
  cen=[]
  array=n.zeros( dims )
  for dim in dims:
    cen.append(dim/2)
  return AddPolyCen(array, cen, planes)

####################################################################
def MakePolyCen(dims, center, planes):
  array=n.zeros( dims )
  return AddPolyCen(array, center, planes)

####################################################################
#def AddPolyCen(array, center, planes):
def OldAddPolyCen(array, center, planes):
    dims = array.shape
    for plane in planes:
        if len(plane) > len(dims):
            raise(Exception, "A plane sent to MakePoly is higher demensions than array.")
        
#    array=n.zeros(dims)
    nplanes=len(planes)
#    print planes
#    print nplanes
    #project the r vector of each point onto each plane vector and count the
    #number that are less than the length of the plane vector.  If that number
    #is equal to the number of planes than the point is within the poly.  */
    loopdims=list(dims)
    loopcen=list(center)
    while len(loopdims) <3:
        loopdims.append(1)
        loopcen.append(1)
    for a1 in range(loopdims[0]):
        for a2 in range(loopdims[1]):
            for a3 in range(loopdims[2]):
                np=0;
                for plane in planes:
                    while len(plane)<3:
                        plane=plane+(0,)
                    x=a1-loopcen[0]
                    #x=a1-loopdims[0]/2
                    y=a2-loopcen[1]
                    #y=a2-loopdims[1]/2
                    z=a3-loopcen[2]
                    #z=a3-loopdims[2]/2
                    if( x*plane[0]+y*plane[1]+z*plane[2] <= \
                        plane[0]*plane[0]+ \
                        plane[1]*plane[1]+ \
                        plane[2]*plane[2]):
                        np=np+1

                if(np == nplanes):
                    a=[a1,a2,a3]
                    index=[]
                    for i in range(3):
                        if loopdims[i]>1:
                            index.append(a[i])
                    array[tuple(index)]=1.0
    return array.copy()

def Rotate(matrix, vector):
    pass
    
if __name__=="__main__":
    test='shift'
    vizarray=None
    s=MakePoly( (128,129) ,( (10,0), (-10,0), (0,10), (0,-10), (5,5), (-5,50) ))
    print(s)
    print(s.shape)
#    from scipy.fftpack import fftshift
#    s=fftshift(s)
#    print s
   
    if test=='shift':
      ss=Shift(sf.fftn(sf.fftshift(s)), (5,10)) 
        

    if test=='zeropad':
        print("s shape before", s.shape)
        z=ZeroPad(s, (2,2,1) )
        print("s shape after", s.shape)
        print(z)
        print(z.shape)
        
    if vizarray!=None:
        from CXDViz import CXDViz
        from tvtk.api import tvtk
        viz=CXDViz()
        viz.SetArray(vizarray)
        import mayavi as m
        v=m.mayavi()
        v.open_vtk_data(tvtk.to_vtk(viz.GetImageData()))
        v.master.wait_window()
    
