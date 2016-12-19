//
//  AppDelegate.m
//  EmojiImages
//
//  Created by Nate Parrott on 10/14/16.
//  Copyright © 2016 Nate Parrott. All rights reserved.
//

#import "AppDelegate.h"

@interface AppDelegate ()

@property (weak) IBOutlet NSWindow *window;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_BACKGROUND, 0), ^{
        [self doIt];
    });
}

- (void)doIt {
    NSData *data = [NSData dataWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"emoji" ofType:@"json"]];
    for (NSDictionary *d in [NSJSONSerialization JSONObjectWithData:data options:0 error:nil]) {
        NSString *emoji = d[@"emoji"];
        NSString *desc = d[@"description"];
        if (emoji && desc) {
            CGFloat size = 64;
            NSImage *image = [[NSImage alloc] initWithSize:NSMakeSize(size, size)];
            [image lockFocus];
            NSFont *font = [NSFont systemFontOfSize:60];
            [emoji drawInRect:NSMakeRect(2, 2, size, size) withAttributes:@{NSFontAttributeName: font}];
            
            NSBitmapImageRep *bitmapRep = [[NSBitmapImageRep alloc] initWithFocusedViewRect:NSMakeRect(0, 0, image.size.width, image.size.height)];
            [image unlockFocus];
            
            NSData *data = [bitmapRep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
            NSString *path = @"/Users/nateparrott/Documents/SW/emojinet/emoji";
            [data writeToFile:[path stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.png", desc]] options:0 error:nil];
        }
    }
    NSLog(@"Done");
}

- (void)applicationWillTerminate:(NSNotification *)aNotification {
    // Insert code here to tear down your application
}


@end
