/** Serial sum and population standard deviation of 20 million float64 values. */
public class Task1 {
    static double[] statistics(double[] a) {
        double sum=0;
        for(double x:a) sum+=x;
        double mean=sum/a.length, squared=0;
        for(double x:a) { double d=x-mean; squared+=d*d; }
        return new double[]{sum,Math.sqrt(squared/a.length)};
    }
    public static void main(String[] args) throws Exception {
        Bench.header("TASK 1 - Hardware probe and serial baseline");
        long total=System.nanoTime(), gen=System.nanoTime();
        double[] a=Bench.input(20_000_000);
        double generation=Bench.ms(gen);
        long s=System.nanoTime(); double[] result=statistics(a); double elapsed=Bench.ms(s); double totalElapsed=Bench.ms(total);
        System.out.println("Elements: " + a.length + "; type: double (float64); application workers: 1");
        System.out.printf("Sum: %.12f%nPopulation standard deviation (ddof=0): %.12f%n",result[0],result[1]);
        System.out.printf("Array generation: %.6f ms%nSerial computation: %.6f ms%nTotal generation + computation: %.6f ms%n",generation,elapsed,totalElapsed);
    }
}
