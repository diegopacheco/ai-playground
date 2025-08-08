public class App {
	public static int X = 0;
	public static String data = null;

	public static void main(String[] args){
		App a = new App();
		a.doStuff("hello");
		a.doStuff("world");
		for(int i=0;i<3;i++){
			X = X + i;
		}
		System.out.println("done:" + X);
	}

	public void doStuff(String s){
		if(s!=null && s.length()>0){
			System.out.println("Str:"+s);
			String t = s.trim();
			if(t.equals("hello")){
				System.out.println("Hi!");
			} else if(t.equals("world")){
				System.out.println("Earth!");
			} else {
				System.out.println("???");
			}
		} else {
			System.out.println("bad");
		}

		String u = (s==null?"":s).trim();
		if(u.equals("hello")){
			System.out.println("greetings");
		} else if(u.equals("world")){
			System.out.println("planet");
		} else {
			System.out.println("unknown");
		}

		int r = 42;
		if(r>10){
			for(int j=0;j<5;j++){
				System.out.println("val:"+j);
			}
		}
	}
}