/*
 * Created by arkd on 4/30/2020.
 * An Optimized but less readable version of Gemm_five_func_loops_packed
 */

#include<immintrin.h>
#include <stdio.h>
#include <stdlib.h>

int MC;
int KC;

#define alpha( a,i,j ) a[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( b,i,j )  b[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( c,i,j ) c[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define min( x, y ) ( ( x ) < ( y ) ? x : y )



static inline void PackMicroPanelA_MRxKC( int m, int k, double *A, int ldA, double *Atilde )
/* Pack a micro-panel of A into buffer pointed to by Atilde.
 * TODO: Use vector intrinsics to speed up*/
{
    /* March through A in column-major order, packing into Atilde as we go. */
    if ( m == MR )   /* Full row size micro-panel.*/
        for ( int p=0; p<k; p++ )
            for ( int i=0; i<MR; i++ )
                *Atilde++ = alpha( i, p );
    else /* Not a full row size micro-panel. TODO: To be implemented, pad w/ zero's */
    {
    }
}
static inline void PackBlockA_MCxKC( int m, int k, double *A, int ldA, double *Atilde )
/* Pack a MC x KC block of A.  MC is assumed to be a multiple of MR.  The block is
   packed into Atilde a micro-panel at a time. If necessary, the last micro-panel
   is padded with rows of zeroes. */
{
    for ( int i=0; i<m; i+= MR ){
        int ib = min( MR, m-i );
        PackMicroPanelA_MRxKC( ib, k, &alpha( i, 0 ), ldA, Atilde );
        Atilde += ib * k;
    }
}
static inline void PackMicroPanelB_KCxNR( int k, int n, double *B, int ldB, double *Btilde )
/* Pack a micro-panel of B into buffer pointed to by Btilde.
   This is an unoptimized implementation for general KC and NR. */
{
    /* March through B in row-major order, packing into Btilde as we go. */

    if ( n == NR ) /* Full column width micro-panel.*/
        for ( int p=0; p<k; p++ )
            for ( int j=0; j<NR; j++ )
                *Btilde++ = beta( p, j );
    else /* Not a full row size micro-panel.  We pad with zeroes. */
        for ( int p=0; p<k; p++ ) {
            for ( int j=0; j<n; j++ )
                *Btilde++ = beta( p, j );
            for ( int j=n; j<NR; j++ )
                *Btilde++ = 0.0;
        }
}

static inline void PackPanelB_KCxNC( int k, int n, double *B, int ldB, double *Btilde )
/* Pack a KC x NC panel of B.  NC is assumed to be a multiple of NR.  The block is
   packed into Btilde a micro-panel at a time. If necessary, the last micro-panel
   is padded with columns of zeroes.
   TODO: use vector intrinsics*/
{
    for ( int j=0; j<n; j+= NR ){
        int jb = min( NR, n-j );

        PackMicroPanelB_KCxNR( k, jb, &beta( 0, j ), ldB, Btilde );
        Btilde += k * jb;
    }
}
void MyGemm(int m, int n, int k, double *A, int ldA,
            double *B, int ldB, double *C, int ldC)
{
    if ( m % MR != 0 || MC % MR != 0 ){
        printf( "m and MC must be multiples of MR\n" );
        exit( 0 );
    }
    if ( n % NR != 0 || NC % NR != 0 ){
        printf( "n and NC must be multiples of NR\n" );
        exit( 0 );
    }
    /* Optimized packing by using aligned buffers */
     double* Atilde = (double *) _mm_malloc(MC*KC*sizeof(double));
     double* Btilde = (double *) _mm_malloc(KC*NC*sizeof(double));
    /*loop5*/
    for ( int j=0; j<n; j+=NC ) {
        int njb = min(NC, n - j);  /* Last loop may not involve a full block */
        double* Bj = &beta(B,0,j);  /* j column of B */
        double* Cj = &gamma(c,0,j);  /* j column of C*/
        /*loop4*/
        for ( int p=0; p<k; p+=KC ) {
            int kpb = min(KC, k - p);    /* Last loop may not involve a full block */
            PackPanelB_KCxNC(kpb, njb, &beta(Bj, p, 0), ldB, Btilde);
            double *Ap = &alpha(0, p);  /* column partition A */
            /*loop3*/
            for (int i = 0; i < m; i += MC) {
                int mib = min(MC, m - i);    /* Last loop may not involve a full block */
                PackBlockA_MCxKC(ib, pb, &alpha(Ap, i, 0), ldA, Atilde);
                double *Cji = &gamma(Cj, i, 0); /* ith row partion of Cj */
                /* loop2, n, m, & k are replaced by njb, mib, kpb here */
                for ( int j2=0; j2<njb; j2+=NR ) {
                    int jb = min(NR, njb - j2);
                    double *MicroPanelB = &Btilde[j2 * kpb];
                    double *Cjij2 = &gamma(Cji, 0, j2);
                    /*loop1*/
                    for (int i2 = 0; i2 < mib; i2 += MR) {
                        int ib = min(MR, mib - i2);
                        Gemm_MRxNRKernel_Packed(kpb, &Atilde[i2 * kpb],
                            MicroPanelB, &gamma(Cjij2, i2, 0), ldC);
                    }
                }
            }
        }
    }
}