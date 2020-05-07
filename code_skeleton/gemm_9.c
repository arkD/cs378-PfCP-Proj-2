/*
 * Created by arkd on 4/30/2020.
 * An Optimized but less readable version of Gemm_five_func_loops_packed
 */

#include<immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define NC 2016 /* column block size, 5th loop: C & B */
#define KC 224 /* partition block size, 4th loop: A & B, 2nd loop: width A, 1st loop: height B */
#define MC 96 /* column block size, 3rd loop: C & A */
#define MR 12 /* row block size, 2nd loop: A, 1st loop: C */
#define NR 4 /* column block size, 2nd loop: C & B, 1st loop: C */

#define alpha( a,i,j ) a[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( b,i,j )  b[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( c,i,j ) c[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define min( x, y ) ( ( x ) < ( y ) ? x : y )

static inline void Gemm_MRxNRKernel_Packed( int, double *, double *, double *, int );
static inline void PackBlockA_MCxKC( int, int, double *, int, double * );
static inline void PackPanelB_KCxNC( int, int, double *, int, double * );

void MyGemm( int m, int n, int k, double *A, int ldA,
             double *B, int ldB, double *C, int ldC )
{
    if ( m % MR != 0 || MC % MR != 0 ){
        printf( "m and MC must be multiples of MR\n" );
        exit( 0 );
    }
    if ( n % NR != 0 || NC % NR != 0 ){
        printf( "n and NC must be multiples of NR\n" );
        exit( 0 );
    }

    double *restrict Atilde = ( double * ) _mm_malloc( MC * KC * sizeof( double ), 64 );
    double *restrict Btilde = ( double * ) _mm_malloc( KC * NC * sizeof( double ), 64 );

    for ( int j=0; j<n; j+=NC ) {
        int njb = min( NC, n-j );    /* Last loop may not involve a full block */
        double *restrict Bj = &beta(B, 0,j );
        double *restrict Cj = &gamma( C,0,j );
        //        LoopFour( m, njb, k, A, ldA, &beta(B, 0,j ), ldB, &gamma( C,0,j ), ldC );

        for ( int p=0; p<k; p+=KC ) {
            int kpb = min( KC, k-p );    /* Last loop may not involve a full block */
            PackPanelB_KCxNC( kpb, njb, &beta( Bj, p, 0 ), ldB, Btilde );
            double *restrict Ap = &alpha(A,0,p);
            //        LoopThree( m, njb, kpb, &alpha( A, 0, p ), ldA, Btilde, Cj, ldC );

            for ( int i=0; i<m; i+=MC ) {
                int mib = min( MC, m-i );    /* Last loop may not involve a full block */
                PackBlockA_MCxKC( mib, kpb, &alpha( Ap, i, 0 ), ldA, Atilde );
                double *restrict Cji = &gamma( Cj,i,0 );
                //        LoopTwo( mib, njb, kpb, Atilde, Btilde, &gamma( Cj,i,0 ), ldC );
                for ( int j2=0; j2<njb; j2+=NR ) {
                    int jb = min( NR, njb-j2 );
                    double *restrict MicroPanelB = &Btilde[ j2*kpb ];
                    double *restrict Cjij2 = &gamma( Cji,0,j2 );
                    //        LoopOne( mib, jb, kpb, Atilde, &Btilde[ j2*kpb ], &gamma( Cji,0,j2 ), ldC );
                    for ( int i2=0; i2<mib; i2+=MR ) {
                        Gemm_MRxNRKernel_Packed( kpb, &Atilde[ i2*kpb ], MicroPanelB, &gamma( Cjij2,i2,0 ), ldC );
                    }
                }
            }
        }
    }
    _mm_free( Atilde);
    _mm_free( Btilde);
}
static inline void PackMicroPanelA_MRxKC( int m, int k, double *A, int ldA, double *Atilde )
/* Pack a micro-panel of A into buffer pointed to by Atilde. */
{
    /* March through A in column-major order, packing into Atilde as we go. */
    __m256d micro_A_4;
    if ( m == MR )   /* Full row size micro-panel.*/
        for ( int p=0; p<k; p++ ) {
            int i = 0;
            for ( ; i<MR-3; i+=4 ) {
                micro_A_4 = _mm256_loadu_pd( &alpha(A, i, p ) );
                _mm256_storeu_pd( Atilde, micro_A_4 );
                Atilde+=4;
            }
            // deal with possibility MR isn't divisible by 4
            for ( ; i<MR; i++)
                *Atilde++ = alpha(A, i, p );
        }
    else /* Not a full row size micro-panel. pad w/ zero's */
    {
        for ( int p=0; p<k; p++ ) {
            int i = 0;
            for ( ; i<m-3; i+=4 ) {
                micro_A_4 = _mm256_loadu_pd( &alpha(A, i, p ) );
                _mm256_storeu_pd( Atilde, micro_A_4 );
                Atilde+=4;
            }
            // deal with remainder
            for ( ; i<m; i++)
                *Atilde++ = alpha(A, i, p );
            // pad with zeros
            for ( ; i<MR; i++)
                *Atilde++ = 0.0;
        }
    }
}

static inline void PackBlockA_MCxKC( int m, int k, double *A, int ldA, double *Atilde )
/* Pack a MC x KC block of A.  MC is assumed to be a multiple of MR.  The block is
   packed into Atilde a micro-panel at a time. If necessary, the last micro-panel
   is padded with rows of zeroes. */
{
    for ( int i=0; i<m; i+= MR ){
        int ib = min( MR, m-i );
        PackMicroPanelA_MRxKC( ib, k, &alpha( A, i, 0 ), ldA, Atilde );
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
                *Btilde++ = beta( B, p, j );
    else /* Not a full row size micro-panel.  We pad with zeroes. */
        for ( int p=0; p<k; p++ ) {
            for ( int j=0; j<n; j++ )
                *Btilde++ = beta( B, p, j );
            for ( int j=n; j<NR; j++ )
                *Btilde++ = 0.0;
        }
}

static inline void PackPanelB_KCxNC( int k, int n, double *B, int ldB, double *Btilde )
/* Pack a KC x NC panel of B.  NC is assumed to be a multiple of NR.  The block is
   packed into Btilde a micro-panel at a time. If necessary, the last micro-panel
   is padded with columns of zeroes. */
{
    for ( int j=0; j<n; j+= NR ){
        int jb = min( NR, n-j );

        PackMicroPanelB_KCxNR( k, jb, &beta( B, 0, j ), ldB, Btilde );
        Btilde += k * jb;
    }
}
/* 12x4 */
static inline void Gemm_MRxNRKernel_Packed( int k, double *MP_A, double *MP_B,
                              double *C, int ldC )
{
    /* Declare vector registers to hold 12x4 C and load them */
    __m256d gamma_0123_0   = _mm256_loadu_pd( &gamma( C,0,0 ) );
    __m256d gamma_0123_1   = _mm256_loadu_pd( &gamma( C,0,1 ) );
    __m256d gamma_0123_2   = _mm256_loadu_pd( &gamma( C,0,2 ) );
    __m256d gamma_0123_3   = _mm256_loadu_pd( &gamma( C,0,3 ) );
    __m256d gamma_4567_0   = _mm256_loadu_pd( &gamma( C,4,0 ) );
    __m256d gamma_4567_1   = _mm256_loadu_pd( &gamma( C,4,1 ) );
    __m256d gamma_4567_2   = _mm256_loadu_pd( &gamma( C,4,2 ) );
    __m256d gamma_4567_3   = _mm256_loadu_pd( &gamma( C,4,3 ) );
    __m256d gamma_891011_0 = _mm256_loadu_pd( &gamma( C,8,0 ) );
    __m256d gamma_891011_1 = _mm256_loadu_pd( &gamma( C,8,1 ) );
    __m256d gamma_891011_2 = _mm256_loadu_pd( &gamma( C,8,2 ) );
    __m256d gamma_891011_3 = _mm256_loadu_pd( &gamma( C,8,3 ) );

    /* Declare vector register for load/broadcasting beta( p,j ) */
    __m256d beta_p_j;
    /* 8 unrolls */
    for ( int p=0; p<k; p+=8 ) {
        /* Declare a vector register to hold the current column of A and load
           it with the four elements of that column. */
        __m256d alpha_0123_p   = _mm256_loadu_pd( MP_A );
        __m256d alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        __m256d alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        /* Load/broadcast beta( p,0 ). update gamma*/
        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
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

        MP_A += MR;
        MP_B += NR;
        /* roll 2 */
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;
        /* roll 3*/
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;

        /* roll 4*/
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;

        /* roll 5*/
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;

        /* roll 6*/
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;

        /* roll 7*/
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;

        /* roll 8*/
        alpha_0123_p   = _mm256_loadu_pd( MP_A );
        alpha_4567_p   = _mm256_loadu_pd( MP_A+4 );
        alpha_891011_p = _mm256_loadu_pd( MP_A+8 );

        beta_p_j = _mm256_broadcast_sd( MP_B );
        gamma_0123_0   = _mm256_fmadd_pd( alpha_0123_p, beta_p_j, gamma_0123_0 );
        gamma_4567_0   = _mm256_fmadd_pd( alpha_4567_p, beta_p_j, gamma_4567_0 );
        gamma_891011_0 = _mm256_fmadd_pd( alpha_891011_p, beta_p_j, gamma_891011_0 );

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

        MP_A += MR;
        MP_B += NR;
    }

    /* Store the updated results */
    _mm256_storeu_pd( &gamma(C,0,0), gamma_0123_0 );
    _mm256_storeu_pd( &gamma(C,4,0), gamma_4567_0 );
    _mm256_storeu_pd( &gamma(C,8,0), gamma_891011_0 );
    _mm256_storeu_pd( &gamma(C,0,1), gamma_0123_1 );
    _mm256_storeu_pd( &gamma(C,4,1), gamma_4567_1 );
    _mm256_storeu_pd( &gamma(C,8,1), gamma_891011_1 );
    _mm256_storeu_pd( &gamma(C,0,2), gamma_0123_2 );
    _mm256_storeu_pd( &gamma(C,4,2), gamma_4567_2 );
    _mm256_storeu_pd( &gamma(C,8,2), gamma_891011_2 );
    _mm256_storeu_pd( &gamma(C,0,3), gamma_0123_3 );
    _mm256_storeu_pd( &gamma(C,4,3), gamma_4567_3 );
    _mm256_storeu_pd( &gamma(C,8,3), gamma_891011_3 );
}