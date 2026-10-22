import java.util.concurrent.*;
public class Task2 {
    public static void main(String[] args) throws Exception {
        Bench.header("TASK 2 - Multi-core scaling vs Amdahl's Law");
        double[] a=Bench.input(20_000_000);
        double expected=Task1.statistics(a)[0], baseline=0;
        System.out.println("Median of 5 runs after 3 warmups; pool creation and input generation excluded; dispatch/join/reduction included.");
        System.out.println("Workers | Wall Time (ms) | Speedup | Amdahl Upper Bound (P=0.95)");
        for(int p:new int[]{1,2,4,Bench.CORES,Bench.CORES*2}) {
            ExecutorService pool=Bench.pool(p); double[] partial=new double[p];
            double elapsed;
            try {
                elapsed=Bench.median(()->{
                    Bench.parallel(pool,p,a.length,w->{
                        int lo=(int)((long)a.length*w/p),hi=(int)((long)a.length*(w+1)/p);
                        double sum=0; for(int i=lo;i<hi;i++) sum+=a[i]; partial[w]=sum;
                    });
                    double sum=0; for(double v:partial) sum+=v; Bench.sink=sum;
                    Bench.check(Math.abs(sum-expected)<=1e-8*Math.abs(expected),"Sum mismatch");
                },3,5);
            } finally { Bench.close(pool); }
            if(p==1) baseline=elapsed;
            System.out.printf("%7d | %14.6f | %7.3f | %10.3f%n",p,elapsed,baseline/elapsed,1/(0.05+0.95/p));
        }
        System.out.println("T(1) is the one-worker sum benchmark above; Task 1 also computes standard deviation and is not its baseline.");
        System.out.println("Amdahl values are model bounds, not guaranteed measured limits; memory, scheduling and CPU heterogeneity affect results.");
    }
}
