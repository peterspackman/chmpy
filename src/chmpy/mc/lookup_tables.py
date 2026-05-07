"""Marching cubes lookup table.

Only the classic Marching Cubes case table (Lorensen 1987) is kept here —
that's all the chmpy numpy MC implementation in _mc_numpy.py consumes.
The Lewiner topological-correctness tables (TILING_*, TEST*, SUBCONFIG13)
that the previous cython port used were stripped along with the cython
extension; if/when we want to disambiguate saddle cases we'll pull them
back in from the upstream scikit-image source.

The base64 string below decodes to a 256x16 int8 array; entry [case_id]
gives up to 5 triangles (3 edge ids each), -1 padded.
"""

CASESCLASSIC = (
    (256, 16),
    """
/////////////////////wAIA/////////////////8AAQn/////////////////AQgDCQgB////
/////////wECCv////////////////8ACAMBAgr/////////////CQIKAAIJ/////////////wII
AwIKCAoJCP////////8DCwL/////////////////AAsCCAsA/////////////wEJAAIDC///////
//////8BCwIBCQsJCAv/////////AwoBCwoD/////////////wAKAQAICggLCv////////8DCQAD
CwkLCgn/////////CQgKCggL/////////////wQHCP////////////////8EAwAHAwT/////////
////AAEJCAQH/////////////wQBCQQHAQcDAf////////8BAgoIBAf/////////////AwQHAwAE
AQIK/////////wkCCgkAAggEB/////////8CCgkCCQcCBwMHCQT/////CAQHAwsC////////////
/wsEBwsCBAIABP////////8JAAEIBAcCAwv/////////BAcLCQQLCQsCCQIB/////wMKAQMLCgcI
BP////////8BCwoBBAsBAAQHCwT/////BAcICQALCQsKCwAD/////wQHCwQLCQkLCv////////8J
BQT/////////////////CQUEAAgD/////////////wAFBAEFAP////////////8IBQQIAwUDAQX/
////////AQIKCQUE/////////////wMACAECCgQJBf////////8FAgoFBAIEAAL/////////AgoF
AwIFAwUEAwQI/////wkFBAIDC/////////////8ACwIACAsECQX/////////AAUEAAEFAgML////
/////wIBBQIFCAIICwQIBf////8KAwsKAQMJBQT/////////BAkFAAgBCAoBCAsK/////wUEAAUA
CwULCgsAA/////8FBAgFCAoKCAv/////////CQcIBQcJ/////////////wkDAAkFAwUHA///////
//8ABwgAAQcBBQf/////////AQUDAwUH/////////////wkHCAkFBwoBAv////////8KAQIJBQAF
AwAFBwP/////CAACCAIFCAUHCgUC/////wIKBQIFAwMFB/////////8HCQUHCAkDCwL/////////
CQUHCQcCCQIAAgcL/////wIDCwABCAEHCAEFB/////8LAgELAQcHAQX/////////CQUICAUHCgED
CgML/////wUHAAUACQcLAAEACgsKAP8LCgALAAMKBQAIAAcFBwD/CwoFBwsF/////////////woG
Bf////////////////8ACAMFCgb/////////////CQABBQoG/////////////wEIAwEJCAUKBv//
//////8BBgUCBgH/////////////AQYFAQIGAwAI/////////wkGBQkABgACBv////////8FCQgF
CAIFAgYDAgj/////AgMLCgYF/////////////wsACAsCAAoGBf////////8AAQkCAwsFCgb/////
////BQoGAQkCCQsCCQgL/////wYDCwYFAwUBA/////////8ACAsACwUABQEFCwb/////AwsGAAMG
AAYFAAUJ/////wYFCQYJCwsJCP////////8FCgYEBwj/////////////BAMABAcDBgUK////////
/wEJAAUKBggEB/////////8KBgUBCQcBBwMHCQT/////BgECBgUBBAcI/////////wECBQUCBgMA
BAMEB/////8IBAcJAAUABgUAAgb/////BwMJBwkEAwIJBQkGAgYJ/wMLAgcIBAoGBf////////8F
CgYEBwIEAgACBwv/////AAEJBAcIAgMLBQoG/////wkCAQkLAgkECwcLBAUKBv8IBAcDCwUDBQEF
Cwb/////BQELBQsGAQALBwsEAAQL/wAFCQAGBQADBgsGAwgEB/8GBQkGCQsEBwkHCwn/////CgQJ
BgQK/////////////wQKBgQJCgAIA/////////8KAAEKBgAGBAD/////////CAMBCAEGCAYEBgEK
/////wEECQECBAIGBP////////8DAAgBAgkCBAkCBgT/////AAIEBAIG/////////////wgDAggC
BAQCBv////////8KBAkKBgQLAgP/////////AAgCAggLBAkKBAoG/////wMLAgABBgAGBAYBCv//
//8GBAEGAQoECAECAQsICwH/CQYECQMGCQEDCwYD/////wgLAQgBAAsGAQkBBAYEAf8DCwYDBgAA
BgT/////////BgQICwYI/////////////wcKBgcICggJCv////////8ABwMACgcACQoGBwr/////
CgYHAQoHAQcIAQgA/////woGBwoHAQEHA/////////8BAgYBBggBCAkIBgf/////AgYJAgkBBgcJ
AAkDBwMJ/wcIAAcABgYAAv////////8HAwIGBwL/////////////AgMLCgYICggJCAYH/////wIA
BwIHCwAJBwYHCgkKB/8BCAABBwgBCgcGBwoCAwv/CwIBCwEHCgYBBgcB/////wgJBggGBwkBBgsG
AwEDBv8ACQELBgf/////////////BwgABwAGAwsACwYA/////wcLBv////////////////8HBgv/
////////////////AwAICwcG/////////////wABCQsHBv////////////8IAQkIAwELBwb/////
////CgECBgsH/////////////wECCgMACAYLB/////////8CCQACCgkGCwf/////////BgsHAgoD
CggDCgkI/////wcCAwYCB/////////////8HAAgHBgAGAgD/////////AgcGAgMHAAEJ////////
/wEGAgEIBgEJCAgHBv////8KBwYKAQcBAwf/////////CgcGAQcKAQgHAQAI/////wADBwAHCgAK
CQYKB/////8HBgoHCggICgn/////////BggECwgG/////////////wMGCwMABgAEBv////////8I
BgsIBAYJAAH/////////CQQGCQYDCQMBCwMG/////wYIBAYLCAIKAf////////8BAgoDAAsABgsA
BAb/////BAsIBAYLAAIJAgoJ/////woJAwoDAgkEAwsDBgQGA/8IAgMIBAIEBgL/////////AAQC
BAYC/////////////wEJAAIDBAIEBgQDCP////8BCQQBBAICBAb/////////CAEDCAYBCAQGBgoB
/////woBAAoABgYABP////////8EBgMEAwgGCgMAAwkKCQP/CgkEBgoE/////////////wQJBQcG
C/////////////8ACAMECQULBwb/////////BQABBQQABwYL/////////wsHBggDBAMFBAMBBf//
//8JBQQKAQIHBgv/////////BgsHAQIKAAgDBAkF/////wcGCwUECgQCCgQAAv////8DBAgDBQQD
AgUKBQILBwb/BwIDBwYCBQQJ/////////wkFBAAIBgAGAgYIB/////8DBgIDBwYBBQAFBAD/////
BgIIBggHAgEIBAgFAQUI/wkFBAoBBgEHBgEDB/////8BBgoBBwYBAAcIBwAJBQT/BAAKBAoFAAMK
BgoHAwcK/wcGCgcKCAUECgQICv////8GCQUGCwkLCAn/////////AwYLAAYDAAUGAAkF/////wAL
CAAFCwABBQUGC/////8GCwMGAwUFAwH/////////AQIKCQULCQsICwUG/////wALAwAGCwAJBgUG
CQECCv8LCAULBQYIAAUKBQIAAgX/BgsDBgMFAgoDCgUD/////wUICQUCCAUGAgMIAv////8JBQYJ
BgAABgL/////////AQUIAQgABQYIAwgCBgII/wEFBgIBBv////////////8BAwYBBgoDCAYFBgkI
CQb/CgEACgAGCQUABQYA/////wADCAUGCv////////////8KBQb/////////////////CwUKBwUL
/////////////wsFCgsHBQgDAP////////8FCwcFCgsBCQD/////////CgcFCgsHCQgBCAMB////
/wsBAgsHAQcFAf////////8ACAMBAgcBBwUHAgv/////CQcFCQIHCQACAgsH/////wcFAgcCCwUJ
AgMCCAkIAv8CBQoCAwUDBwX/////////CAIACAUCCAcFCgIF/////wkAAQUKAwUDBwMKAv////8J
CAIJAgEIBwIKAgUHBQL/AQMFAwcF/////////////wAIBwAHAQEHBf////////8JAAMJAwUFAwf/
////////CQgHBQkH/////////////wUIBAUKCAoLCP////////8FAAQFCwAFCgsLAwD/////AAEJ
CAQKCAoLCgQF/////woLBAoEBQsDBAkEAQMBBP8CBQECCAUCCwgEBQj/////AAQLAAsDBAULAgsB
BQEL/wACBQAFCQILBQQFCAsIBf8JBAUCCwP/////////////AgUKAwUCAwQFAwgE/////wUKAgUC
BAQCAP////////8DCgIDBQoDCAUEBQgAAQn/BQoCBQIEAQkCCQQC/////wgEBQgFAwMFAf//////
//8ABAUBAAX/////////////CAQFCAUDCQAFAAMF/////wkEBf////////////////8ECwcECQsJ
Cgv/////////AAgDBAkHCQsHCQoL/////wEKCwELBAEEAAcEC/////8DAQQDBAgBCgQHBAsKCwT/
BAsHCQsECQILCQEC/////wkHBAkLBwkBCwILAQAIA/8LBwQLBAICBAD/////////CwcECwQCCAME
AwIE/////wIJCgIHCQIDBwcECf////8JCgcJBwQKAgcIBwACAAf/AwcKAwoCBwQKAQoABAAK/wEK
AggHBP////////////8ECQEEAQcHAQP/////////BAkBBAEHAAgBCAcB/////wQAAwcEA///////
//////8ECAf/////////////////CQoICgsI/////////////wMACQMJCwsJCv////////8AAQoA
CggICgv/////////AwEKCwMK/////////////wECCwELCQkLCP////////8DAAkDCQsBAgkCCwn/
////AAILCAAL/////////////wMCC/////////////////8CAwgCCAoKCAn/////////CQoCAAkC
/////////////wIDCAIICgABCAEKCP////8BCgL/////////////////AQMICQEI////////////
/wAJAf////////////////8AAwj//////////////////////////////////////w==
""",
)
