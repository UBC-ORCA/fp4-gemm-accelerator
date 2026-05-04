#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#define i32 int32_t
#define u32 uint32_t
#define i16 uint16_t
#define u16 uint16_t
#define i8  uint8_t
#define u8  uint8_t


#define N 8

#define F 3
#define pF (F/2)
#define nF (-pF)


i32 A[N*N];
i32 B[N*N];
i32 C[N*N];

i32 FILT[F*F];



void initM( i32 *AA, u32 D )
{
        for( i32 i=0; i<D*D; i++ )
                AA[i] = rand() & 0x07;
}


void printM( i32 *AA, u32 D )
{
        for( i32 r=0; r<D; r++ ) {
                for( i32 c=0; c<D; c++ ) {
                        printf( "%02x ", AA[ r*D + c ] );
                }
                printf( "\n" );
        }

}

void conv_traditional( i32 *S, i32 *R )
{
        for( i32 r=0; r<N; r++ ) {
                for( i32 c=0; c<N; c++ ) {
                        i32 sum = 0;
                        for( i32 x=0; x<F; x++ ) {
                                for( i32 y=0; y<F; y++ ) {
                                        if( (r+y < N) && (c+x < N) )
                                                sum += S[ (r+y)*N+(c+x) ] * FILT[ y*F + x ];
                                }
                        }
                        R[ r*N + c ] = sum;
                }
        }
}

void conv_mac64( i32 *S, i32 *R )
{
        for( i32 x=0; x<F; x++ ) {
                for( i32 y=0; y<F; y++ ) {

                        i8 f = FILT[ y*F + x ];

                        // this part can be implemented in one instruction over the MAC64 array
                        for( i32 r=0; r<N; r++ ) {
                                for( i32 c=0; c<N; c++ ) {
                                        if( (r+y < N) && (c+x < N) )
                                                R[ r*N + c ] += f * S[ (r+y)*N + c+x ];
                                }
                        }

                }
        }
}

int main()
{
        assert( (pF-nF+1) == F );

        initM( A, N );
        initM( FILT, F );

        printM( A, N ); printf("\n");
        printM( FILT, F ); printf("\n");
        conv_traditional( B );
        conv_mac64( C );
        printM( A, N ); printf("\n");
        printM( B, N ); printf("\n");
        printM( C, N ); printf("\n");

        conv2();

        printf( "Hello, world!\n" );
        return 0;
}
