
#ifndef __GAGE_MULTIPLESYSTEM_H__
#define __GAGE_MULTIPLESYSTEM_H__

/*	
	Define to support Windows 2000 and higher
*/

typedef struct __THREADSTRUCT
{
	CSHANDLE	hSystem;
	TCHAR		lpFilename[100];
	int			Sys;
}THREADSTRUCT;


DWORD WINAPI MultipleSystemThreadFunc(void *arg);


#endif // __GAGE_MULTIPLESYSTEM_H__