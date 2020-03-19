import torch
from torchsearchsorted import searchsorted, numpy_searchsorted
import time

if __name__ == '__main__':
    # defining the number of tests
    ntests = 2

    # defining the problem dimensions
    nrows_a = 50000
    nrows_v = 50000
    nsorted_values = 300
    nvalues = 1000

    # defines the variables. The first run will comprise allocation, the
    # further ones will not
    test_GPU = None
    test_CPU = None

    for ntest in range(ntests):
        print("\nLooking for %dx%d values in %dx%d entries" % (nrows_v, nvalues,
                                                             nrows_a,
                                                             nsorted_values))

        side = 'right'
        # generate a matrix with sorted rows
        a = torch.randn(nrows_a, nsorted_values, device='cpu')
        a = torch.sort(a, dim=1)[0]
        # generate a matrix of values to searchsort
        v = torch.randn(nrows_v, nvalues, device='cpu')

        # a = torch.tensor([[0., 1.]])
        # v = torch.tensor([[1.]])

        t0 = time.time()
        test_NP = torch.tensor(numpy_searchsorted(a, v, side))
        print('NUMPY:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))
        t0 = time.time()
        test_CPU = searchsorted(a, v, test_CPU, side)
        print('CPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))
        # compute the difference between both
        error_CPU = torch.norm(test_NP.double()
                               - test_CPU.double()).numpy()
        if error_CPU:
            import ipdb; ipdb.set_trace()
        print('    difference between CPU and NUMPY: %0.3f' % error_CPU)

        if not torch.cuda.is_available():
            print('CUDA is not available on this machine, cannot go further.')
            continue
        else:
            # now do the CPU
            a = a.to('cuda')
            v = v.to('cuda')
            torch.cuda.synchronize()
            # launch searchsorted on those
            t0 = time.time()
            test_GPU = searchsorted(a, v, test_GPU, side)
            torch.cuda.synchronize()
            print('GPU:  searchsorted in %0.3fms' % (1000*(time.time()-t0)))

            # compute the difference between both
            error_CUDA = torch.norm(test_NP.to('cuda').double()
                               - test_GPU.double()).cpu().numpy()

            print('    difference between GPU and NUMPY: %0.3f' % error_CUDA)
