/**
 * @file MarginalOrder.cpp
 * @author Daniel Sharp
 * @brief This file contains the implementation of the CreateMarginalOrder function
 * @version 0.1
 * @date 2024-02-22
 * 
 * This is a sample implementation of CreateMarginalOrder function, which isn't
 * included in the stock MParT. While one can make these fixed multi-index sets
 * via stock MParT, it is highly inefficient and time-consuming for high-dimensions
 * and high orders.
 * 
 * This can easily be bound to Julia or Python via the wrapping mechanisms
 * modeled in MParT/bindings/python/src and MParT/bindings/julia/src
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <Kokkos_Core.hpp>
#include <MParT/FixedMultiIndexSet.hpp>
#include <MParT/Utilities/KokkosSpaceMappings.h>

using MemorySpace = Kokkos::HostSpace;

FixedMultiIndexSet<MemorySpace> FixedMultiIndexSet<MemorySpace>::CreateMarginalOrder(unsigned int _dim,
    unsigned int _order, int _whichDim) {
    // constants Setup
    unsigned int numTerms = _whichDim < 0 ? _dim*_order+1 : _order;
    unsigned int numNz = numTerms - int(_whichDim < 0);
    // Setup the views for fmset
    Kokkos::View<unsigned int*, MemorySpace> nzStarts("Start of a Multiindex", numTerms+1);
    Kokkos::View<unsigned int*, MemorySpace> nzDims("Nonzero dims", numNz);
    Kokkos::View<unsigned int*, MemorySpace> nzOrders("Nonzero orders", numNz);
    if(_whichDim < 0) { // If we want all the dimensions
        Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(1, numTerms);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(unsigned int i) {
            unsigned int term_start = i-1;
            nzStarts(i) = term_start;
            nzDims(term_start) = i%_dim;
            nzOrders(term_start) = (i+_dim-1)/_dim;
            if (i == numTerms-1) {
                nzStarts(numTerms) = numTerms-1;
            }
        });
    } else { // If we only want nonzeros in a specific dimension
        Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0, numTerms);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(unsigned int i) {
            nzStarts(i) = i;
            nzDims(i) = _whichDim;
            nzOrders(i) = i+1;
            if (i == numTerms-1) {
                nzStarts(numTerms) = numTerms;
            }
        });
    }

    return FixedMultiIndexSet<MemorySpace>(_dim, nzStarts, nzDims, nzOrders);
}