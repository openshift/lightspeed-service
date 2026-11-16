#!/usr/bin/env perl
# Run a command with a deadline, terminating its entire process group on timeout.
use strict;
use warnings;
use POSIX qw(WIFEXITED WEXITSTATUS WIFSIGNALED WTERMSIG ceil
    SIGINT SIGTERM SIGHUP SIG_BLOCK SIG_UNBLOCK);

# GNU timeout accepts fractional durations. Time::HiRes keeps them exact, but it
# is a separate package on RHEL, so a perl without it must still enforce the
# deadline - rounded up to whole seconds - instead of dying at compile time.
my $HAVE_HIRES = eval { require Time::HiRes; 1 } ? 1 : 0;

sub start_deadline {
    my ($seconds) = @_;
    return if $seconds <= 0;
    if ($HAVE_HIRES) {
        Time::HiRes::alarm($seconds);
        return;
    }
    alarm(ceil($seconds));
}

sub cancel_deadline {
    if ($HAVE_HIRES) {
        Time::HiRes::alarm(0);
        return;
    }
    alarm(0);
}

my $seconds = shift @ARGV;
if (!defined($seconds) || $seconds !~ /\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\z/ || !@ARGV) {
    die "usage: run_with_timeout.pl <seconds> <command> [args...]\n" .
        "       <seconds> 0 means no deadline; fractions such as 0.5 are allowed\n";
}
my @command = @ARGV;

# A cancellation signal that lands between fork() and handler installation would
# kill this process with the default disposition, leaving the child - already in
# its own process group - running with no deadline and nobody to stop it. Block
# the cancellation signals across fork and unblock them once the handlers are in
# place.
my $cancel_signals = POSIX::SigSet->new(SIGINT, SIGTERM, SIGHUP);
POSIX::sigprocmask(SIG_BLOCK, $cancel_signals);

pipe(my $ready_read, my $ready_write) or die "pipe: $!\n";
my $pid = fork();
defined($pid) or die "fork: $!\n";

if ($pid == 0) {
    # Restore default signal handling for the command itself; a pending
    # cancellation signal is delivered here, before the process group exists,
    # so it cannot orphan anything.
    POSIX::sigprocmask(SIG_UNBLOCK, $cancel_signals);
    close($ready_read);
    setpgrp(0, 0) or die "setpgrp: $!\n";
    print {$ready_write} "ready\n" or die "write readiness: $!\n";
    close($ready_write) or die "close readiness pipe: $!\n";
    exec {$command[0]} @command or die "exec $command[0]: $!\n";
}

close($ready_write);
my $ready = <$ready_read>;
close($ready_read);

# If process-group setup failed, the child exits without sending readiness.
if (!defined($ready)) {
    waitpid($pid, 0);
    my $status = $?;
    exit WIFEXITED($status) ? WEXITSTATUS($status) : 1;
}

my $stop_command = sub {
    my ($exit_code) = @_;
    cancel_deadline();
    # The child created its own process group before signaling readiness.
    # Signal the group so wrappers such as `uv` cannot leave the actual CLI
    # running after the timeout or cancellation.
    kill 'TERM', -$pid;
    select(undef, undef, undef, 1);
    kill 'KILL', -$pid;
    waitpid($pid, 0);
    exit $exit_code;
};
$SIG{ALRM} = sub { $stop_command->(124); };
$SIG{INT} = sub { $stop_command->(130); };
$SIG{TERM} = sub { $stop_command->(143); };
$SIG{HUP} = sub { $stop_command->(129); };
# Releasing the block delivers a cancellation signal that arrived while masked,
# so the handler above runs instead of the default disposition.
POSIX::sigprocmask(SIG_UNBLOCK, $cancel_signals);
start_deadline($seconds);
waitpid($pid, 0);
my $status = $?;
cancel_deadline();

if (WIFEXITED($status)) {
    exit WEXITSTATUS($status);
}
if (WIFSIGNALED($status)) {
    exit 128 + WTERMSIG($status);
}
exit 1;
