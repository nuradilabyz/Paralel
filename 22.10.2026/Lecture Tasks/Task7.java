import java.util.concurrent.*;
public class Task7 {
    static void kernel(double[] a,double[] b,int lo,int hi,boolean compute){
        if(compute)for(int i=lo;i<hi;i++)b[i]=Math.sin(a[i])*Math.cos(a[i])+Math.tan(a[i]);
        else for(int i=lo;i<hi;i++)b[i]=a[i];
    }
    public static void main(String[] args)throws Exception {
        Bench.header("TASK 7 - Arithmetic intensity versus memory bandwidth");
        int n=30_000_000;double[] a=Bench.input(n),b=new double[n];
        double memoryEfficiency=0,computeEfficiency=0;
        System.out.println("30,000,000 double elements; median of 5 runs after 3 warmups. Prestarted pools; dispatch/join included.");
        System.out.println("Kernel | Workers | Wall Time (ms) | Speedup | Efficiency | Effective GB/s (copy only)");
        for(boolean compute:new boolean[]{false,true}) {
            double baseline=0;
            for(int p:new int[]{1,2,Bench.CORES}) {
                ExecutorService pool=Bench.pool(p);double t;
                try {
                    t=Bench.median(()->{
                        Bench.parallel(pool,p,n,w->kernel(a,b,(int)((long)n*w/p),(int)((long)n*(w+1)/p),compute));
                        Bench.sink=b[n-1];
                    },3,5);
                }finally{Bench.close(pool);}
                for(int i=0;i<n;i++) {
                    double expected=compute?Math.sin(a[i])*Math.cos(a[i])+Math.tan(a[i]):a[i];
                    Bench.check(Math.abs(b[i]-expected)<=1e-12,"Kernel mismatch");
                }
                if(p==1)baseline=t;double speedup=baseline/t,eff=speedup/p;
                if(p==Bench.CORES){if(compute)computeEfficiency=eff;else memoryEfficiency=eff;}
                System.out.printf("%s | %d | %.6f | %.3f | %.3f | %s%n",compute?"Compute":"Copy",p,t,speedup,eff,compute?"n/a":String.format("%.3f",16.0*n/(t*1e6)));
            }
        }
        if(memoryEfficiency<0.75&&memoryEfficiency<computeEfficiency)
            System.out.println("Copy stalled in scaling: lower max-core efficiency than compute is consistent with memory-channel saturation. This is a timing-based inference; no hardware bandwidth counters were collected.");
        else System.out.println("Timing does not clearly isolate memory-channel saturation in Copy; do not claim a proven stall. Inspect both scaling curves and hardware counters.");
        System.out.println("Copy GB/s counts 8-byte read + 8-byte write per element; cache/write-allocation traffic is not included. Efficiency = T(1) / T(p) / p.");
    }
}
