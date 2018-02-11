package com.newsplore.util;

import java.util.List;

public class StringUtil {
   
	public String join(String sepator,String[] strArr)
	{
		String output = "";
		for(String str:strArr)
		{
			output = output + sepator + str;
		}
		return output;
	}
	
	public String join(String sepator,List<String> strArr)
	{
		String output = "";
		for(String str:strArr)
		{
			output = output + sepator + str;
		}
		return output;
	}
}
