//
// Created by arkd on 4/30/2020.
// Copied from assignments/week3
// from LAFF-ON-PfHP

#include <stdio.h>
#include <stdlib.h>

int MC;
int KC;

#define alpha( a, i,j ) a[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A
#define beta( b, i,j )  b[ (j)*ldB + (i) ]   // map beta( i,j ) to array B
#define gamma( c, i,j ) c[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#define min( x, y ) ( ( x ) < ( y ) ? x : y )

void LoopFive( int, int, int, double *, int, double *, int, double *, int );
void LoopFour( int, int, int, double *, int, double *, int,  double *, int );
void LoopThree( int, int, int, double *, int, double *, double *, int );
void LoopTwo( int, int, int, double *, double *, double *, int );
void LoopOne( int, int, int, double *, double *, double *, int );
void Gemm_MRxNRKernel_Packed( int, double *, double *, double *, int );
void PackBlockA_MCxKC( int, int, double *, int, double * );
void PackPanelB_KCxNC( int, int, double *, int, double * );

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

    LoopFive( m, n, k, A, ldA, B, ldB, C, ldC );
}

void LoopFive( int m, int n, int k, double *A, int ldA,
               double *B, int ldB, double *C, int ldC )
{
    for ( int j=0; j<n; j+=NC ) {
        int njb = min( NC, n-j );    /* Last loop may not involve a full block */
        LoopFour( m, njb, k, A, ldA, &beta(B, 0,j ), ldB, &gamma( C,0,j ), ldC );
    }
}

void LoopFour( int m, int njb, int k, double *A, int ldA, double *Bj, int ldB,
               double *Cj, int ldC )
{
    double *Btilde = ( double * ) malloc( KC * NC * sizeof( double ) );

    for ( int p=0; p<k; p+=KC ) {
        int pb = min( KC, k-p );    /* Last loop may not involve a full block */
        PackPanelB_KCxNC( pb, njb, &beta( Bj, p, 0 ), ldB, Btilde );
        LoopThree( m, njb, pb, &alpha( A, 0, p ), ldA, Btilde, Cj, ldC );
    }

    free( Btilde);
}

void LoopThree( int m, int njb, int kpb, double *Ap, int ldA, double *Btilde, double *Cj, int ldC )
{
    double *Atilde = ( double * ) malloc( MC * KC * sizeof( double ) );

    for ( int i=0; i<m; i+=MC ) {
        int mib = min( MC, m-i );    /* Last loop may not involve a full block */
        PackBlockA_MCxKC( mib, kpb, &alpha( Ap, i, 0 ), ldA, Atilde );
        LoopTwo( mib, njb, kpb, Atilde, Btilde, &gamma( Cj,i,0 ), ldC );
    }

    free( Atilde);
}

void LoopTwo( int mib, int njb, int kpb, double *Atilde, double *Btilde, double *Cji, int ldC )
{
    for ( int j2=0; j2<njb; j2+=NR ) {
        int jb = min( NR, njb-j2 );
        LoopOne( mib, jb, kpb, Atilde, &Btilde[ j2*kpb ], &gamma( Cji,0,j2 ), ldC );
    }
}

void LoopOne( int mib, int jb, int kpb, double *Atilde, double *MicroPanelB, double *Cjij2, int ldC )
{
    for ( int i2=0; i2<mib; i2+=MR ) {
        int ib = min( MR, mib-i2 );
        Gemm_MRxNRKernel_Packed( kpb, &Atilde[ i2*kpb ], MicroPanelB, &gamma( Cjij2,i2,0 ), ldC );
    }
}
void PackMicroPanelA_MRxKC( int m, int k, double *A, int ldA, double *Atilde )
/* Pack a micro-panel of A into buffer pointed to by Atilde.
   This is an unoptimized implementation for general MR and KC. */
{
    /* March through A in column-major order, packing into Atilde as we go. */

    if ( m == MR )   /* Full row size micro-panel.*/
        for ( int p=0; p<k; p++ )
            for ( int i=0; i<MR; i++ )
                *Atilde++ = alpha( A, i, p );
    else /* Not a full row size micro-panel.  To be implemented */
    {
    }
}

void PackBlockA_MCxKC( int m, int k, double *A, int ldA, double *Atilde )
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
void PackMicroPanelB_KCxNR( int k, int n, double *B, int ldB, double *Btilde )
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

void PackPanelB_KCxNC( int k, int n, double *B, int ldB, double *Btilde )
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