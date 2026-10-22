import java.util.concurrent.*;
public class Task5 {
    static void affine(double[] x,double[] y,int lo,int hi) {
        for(int i=lo;i<hi;i++) y[i]=2.5*x[i]+1.2;
    }
    public static void main(String[] args) throws Exception {
        Bench.header("TASK 5 - Parallel overhead crossover");
        int p=Bench.CORES;ExecutorService pool=Bench.pool(p);
        int[] sizes={100,1000,10000,100000,1000000,10000000};
        double[] ratios=new double[sizes.length];
        System.out.println("Prestarted reusable pool, all logical cores. Input/allocation/pool startup excluded; dispatch and join included.");
        System.out.println("Median of 9 paired measurements, alternating order, after 5 paired warmups. 10 operations per measurement.");
        System.out.println("N | Serial (ms) | Parallel (ms) | Speedup | Winner");
        try {
            for(int k=0;k<sizes.length;k++) {
                int n=sizes[k];double[] x=Bench.input(n),serial=new double[n],parallel=new double[n];
                Bench.Action a=()->affine(x,serial,0,n);
                Bench.Action b=()->Bench.parallel(pool,p,n,w->affine(x,parallel,(int)((long)n*w/p),(int)((long)n*(w+1)/p)));
                for(int j=0;j<5;j++){a.run();b.run();}
                double[] st=new double[9],pt=new double[9];
                for(int j=0;j<9;j++) {
                    if(j%2==0){st[j]=batch(a);pt[j]=batch(b);}else{pt[j]=batch(b);st[j]=batch(a);}
                }
                java.util.Arrays.sort(st);java.util.Arrays.sort(pt);
                for(int i=0;i<n;i++)Bench.check(serial[i]==parallel[i],"Vector mismatch");
                Bench.sink=parallel[n-1];ratios[k]=st[4]/pt[4];
                System.out.printf("%9d | %.6f | %.6f | %.3f | %s%n",n,st[4],pt[4],ratios[k],ratios[k]>1?"Parallel":"Serial");
            }
        } finally {Bench.close(pool);}
        boolean found=false;
        for(int k=1;k<sizes.length;k++)if(ratios[k-1]<=1&&ratios[k]>1) {
            found=true;System.out.printf("Observed slower-to-faster bracket: %d < N <= %d%n",sizes[k-1],sizes[k]);
        }
        if(!found)System.out.println("No slower-to-faster crossover observed among tested sizes.");
        System.out.println("These are measured brackets, not an exact threshold for untested N; JIT, cache state and noise can change the crossover.");
    }
    static double batch(Bench.Action action)throws Exception {
        long s=System.nanoTime();for(int j=0;j<10;j++)action.run();return Bench.ms(s)/10;
    }
}
