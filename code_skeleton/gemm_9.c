/*
 * Created by arkd on 4/30/2020.
 * An Optimized but less readable version of Gemm_five_func_loops_packed
 */

#include<immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define MC -1
#define MR -1
#define NC -1
#define NR -1
#define KC -1

#define alpha( a,i,j ) a[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( b,i,j )  b[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( c,i,j ) c[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define min( x, y ) ( ( x ) < ( y ) ? x : y )

/* 12x4 */
static inline void Gemm_MRxNRKernel_Packed( int k,
                                            double *MP_A, double *MP_B, double *C, int ldC )
{
    /* Declare vector registers to hold 4x4 C and load them */
    __m256d gamma_0123_0 = _mm256_loadu_pd( &gamma( C,0,0 ) );
    __m256d gamma_0123_1 = _mm256_loadu_pd( &gamma( C,0,1 ) );
    __m256d gamma_0123_2 = _mm256_loadu_pd( &gamma( C,0,2 ) );
    __m256d gamma_0123_3 = _mm256_loadu_pd( &gamma( C,0,3 ) );

    __m256d gamma_4567_0 = _mm256_loadu_pd( &gamma( C,4,0 ) );
    __m256d gamma_4567_1 = _mm256_loadu_pd( &gamma( C,4,1 ) );
    __m256d gamma_4567_2 = _mm256_loadu_pd( &gamma( C,4,2 ) );
    __m256d gamma_4567_3 = _mm256_loadu_pd( &gamma( C,4,3 ) );

    __m256d gamma_891011_0 = _mm256_loadu_pd( &gamma( C,8,0 ) );
    __m256d gamma_891011_1 = _mm256_loadu_pd( &gamma( C,8,1 ) );
    __m256d gamma_891011_2 = _mm256_loadu_pd( &gamma( C,8,2 ) );
    __m256d gamma_891011_3 = _mm256_loadu_pd( &gamma( C,8,3 ) );
    for ( int p=0; p<k; p++ ){
        /* Declare vector register for load/broadcasting beta( p,j ) */
        __m256d beta_p_j;

        /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
        __m256d alpha_0123_p = _mm256_loadu_pd( MP_A );
        __m256d alpha_4567_p = _mm256_loadu_pd( MP_A+4 ) );
        __m256d alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        /* Load/broadcast beta( p,0 ). update gamma*/
        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

        /* REPEAT for second, third, and fourth columns of C.  Notice that the
           current column of A needs not be reloaded. */
        beta_p_j = _mm256_broadcast_sd( MP_B+1 );
        gamma_0123_1 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_1 );
        gamma_4567_1 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_1 );
        gamma_891011_1 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_1 );

        beta_p_j = _mm256_broadcast_sd( MP_B+2 );
        gamma_0123_2 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_2 );
        gamma_4567_2 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_2 );
        gamma_891011_2 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_2 );

        beta_p_j = _mm256_broadcast_sd( MP_B+3 );
        gamma_0123_3 = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_3 );
        gamma_4567_3 = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_3 );
        gamma_891011_3 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_3 );

    }

    /* Store the updated results */
    _mm256_storeu_pd( &gamma(C,0,0), gamma_0123_0 );
    _mm256_storeu_pd( &gamma(C,0,1), gamma_0123_1 );
    _mm256_storeu_pd( &gamma(C,0,2), gamma_0123_2 );
    _mm256_storeu_pd( &gamma(C,0,3), gamma_0123_3 );

    _mm256_storeu_pd( &gamma(C,4,0), gamma_4567_0 );
    _mm256_storeu_pd( &gamma(C,4,1), gamma_4567_1 );
    _mm256_storeu_pd( &gamma(C,4,2), gamma_4567_2 );
    _mm256_storeu_pd( &gamma(C,4,3), gamma_4567_3 );

    _mm256_storeu_pd( &gamma(C,8,0), gamma_891011_0 );
    _mm256_storeu_pd( &gamma(C,8,1), gamma_891011_1 );
    _mm256_storeu_pd( &gamma(C,8,2), gamma_891011_2 );
    _mm256_storeu_pd( &gamma(C,8,3), gamma_891011_3 );
}

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
    double* Atilde = (double *) _mm_malloc(MC*KC*sizeof(double), 64);
    double* Btilde = (double *) _mm_malloc(KC*NC*sizeof(double), 64);
    /*loop5*/
    for ( int j=0; j<n; j+=NC ) {
        int njb = min(NC, n - j);  /* Last loop may not involve a full block */
        double *restrict Bj = &beta(B,0,j);  /* j column of B */
        double *restrict Cj = &gamma(c,0,j);  /* j column of C*/
        /*loop4*/
        for ( int p=0; p<k; p+=KC ) {
            int kpb = min(KC, k - p);    /* Last loop may not involve a full block */
            PackPanelB_KCxNC(kpb, njb, &beta(Bj, p, 0), ldB, Btilde);
            double *restrict Ap = &alpha(0, p);  /* column partition A */
            /*loop3*/
            for (int i = 0; i < m; i += MC) {
                int mib = min(MC, m - i);    /* Last loop may not involve a full block */
                PackBlockA_MCxKC(ib, pb, &alpha(Ap, i, 0), ldA, Atilde);
                double *restrict Cji = &gamma(Cj, i, 0); /* ith row partion of Cj */
                /* loop2, n, m, & k are replaced by njb, mib, kpb here */
                for ( int j2=0; j2<njb; j2+=NR ) {
                    int jb = min(NR, njb - j2);
                    double *restrict MicroPanelB = &Btilde[j2 * kpb];
                    double *restrict Cjij2 = &gamma(Cji, 0, j2);
                    /*loop1*/
                    for (int i2 = 0; i2 < mib; i2 += MR) {
                        int ib = min(MR, mib - i2); /* unused?? */
                        Gemm_MRxNRKernel_Packed(kpb, &Atilde[i2 * kpb],
                            MicroPanelB, &gamma(Cjij2, i2, 0), ldC);
                    }
                }
            }
        }
    }
    _mm_free(Atilde);
    _mm_free(Btilde);
}