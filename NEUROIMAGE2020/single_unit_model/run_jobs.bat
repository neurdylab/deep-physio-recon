
SET startpoint=1
SET endpoint=2

echo off
echo "Starting the runs " 
set startpath=%CD% 
set "backsl=\"

set pathnospace=%startpath: =%
echo "Starting at %startpath% "
echo "nospace %pathnospace%"
FOR %%G IN (run0, run1, run2, run3, run4, run5, run6, run7, run8, run9, run10, run11, run12, run13, run14, run15, run16, run17, run18, run19, run20, run21, run22, run23, run24, run25, run26, run27, run28, run29, run30, run31, run32, run33, run34, run35, run36, run37, run38, run39, run40, run41, run42, run43, run44, run45, run46, run47, run48, run49, run50, run51, run52, run53, run54, run55, run56 ) DO  (
    
   
   echo "Going in %%G";     

   

   echo "Full path for folder %%G is"
   cd %%G
   echo "After change"
   echo %CD%
   echo %%G >> location.dat 
   echo %CD% >> location.dat 
   echo "Inside of %%G" 
   python model_single.py   
   cd ..
   echo "Out of %%G" 
)
echo "Ended" 

echo Back at the start point 
echo %CD%


