//判断数组aa中是否存在指定元素a
String a = "a";
String[] aa = {"a","b"};
List<String>  aaa = Arrays.asList(aa); //需要先将数组aa转为list型。另不能在asList函数中直接用{"a","b"}初始化，但是可以用new String[]{"a","b"}

if (aaa.stream().anyMatch( p -> p.equals(a))){   // aaa中任何一个元素与a相同
    System.out.println("yes");
}
